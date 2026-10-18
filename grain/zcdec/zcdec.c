/*
 * zcdec: decode a video stream on VA-API and hand the kept frames to another process as DMA-BUF file descriptors,
 * so grain_apply.py can import the decoder's own surfaces into torch's Level Zero context and read them on the card
 * (no hwdownload, no 25 MB of pixels per frame through system memory).
 *
 *   zcdec SRC SOCKFD STEP [A:B ...]
 *
 * The first video stream of SRC (ffmpeg's 0:v:0) is decoded in output order; frame n is kept when STEP > 0 and
 * n % STEP == 0, or when A <= n < B for one of the ranges - grain_apply's keep(). For each kept frame one message
 * goes over the SOCK_SEQPACKET socket SOCKFD: a struct zc_msg plus the frame's DMA-BUF fds (SCM_RIGHTS). The surface
 * stays referenced until the reader acks it with one byte, in order; at most ZC_INFLIGHT (env, default 4) are out at
 * a time. A message with n = -1 ends the stream (pts = frames decoded), after which the reader acks what it still
 * holds; n = -2 reports a failure (text in err).
 */
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_drm.h>
#include <libavutil/pixdesc.h>

#define ZC_MAX_OBJ 4
#define ZC_MAX_LAYERS 4
#define ZC_MAX_RANGES 64

struct zc_layer {
    uint32_t format;          /* DRM fourcc of the layer (R16, GR1616, R8, GR88) */
    int32_t nb_planes;
    int32_t object[4];
    int64_t offset[4];
    int64_t pitch[4];
};

struct zc_msg {
    int32_t n;                /* frame number in output order; -1 end, -2 error */
    int32_t width, height;
    int32_t sw_format;        /* surface format: 1 P010, 2 NV12, 0 anything else */
    int32_t bits;             /* component depth */
    int32_t nb_objects;
    uint64_t size[ZC_MAX_OBJ];
    uint64_t modifier[ZC_MAX_OBJ];
    int32_t nb_layers;
    struct zc_layer layer[ZC_MAX_LAYERS];
    int64_t pts;
    char err[160];
};

static int sock = -1, step = 0, nranges = 0;
static int64_t ra[ZC_MAX_RANGES], rb[ZC_MAX_RANGES];

static int kept(int64_t n)
{
    if (step > 0 && n % step == 0)
        return 1;
    for (int i = 0; i < nranges; i++)
        if (n >= ra[i] && n < rb[i])
            return 1;
    return 0;
}

static int send_msg(const struct zc_msg *m, const int *fds, int nfds)
{
    struct iovec iov = {.iov_base = (void *)m, .iov_len = sizeof(*m)};
    char cbuf[CMSG_SPACE(sizeof(int) * ZC_MAX_OBJ)];
    struct msghdr mh = {.msg_iov = &iov, .msg_iovlen = 1};
    if (nfds > 0) {
        memset(cbuf, 0, sizeof(cbuf));
        mh.msg_control = cbuf;
        mh.msg_controllen = CMSG_SPACE(sizeof(int) * nfds);
        struct cmsghdr *c = CMSG_FIRSTHDR(&mh);
        c->cmsg_level = SOL_SOCKET;
        c->cmsg_type = SCM_RIGHTS;
        c->cmsg_len = CMSG_LEN(sizeof(int) * nfds);
        memcpy(CMSG_DATA(c), fds, sizeof(int) * nfds);
    }
    for (;;) {
        ssize_t r = sendmsg(sock, &mh, MSG_NOSIGNAL);
        if (r == (ssize_t)sizeof(*m))
            return 0;
        if (r < 0 && errno == EINTR)
            continue;
        return -1;
    }
}

static void fail(const char *what, int err)
{
    struct zc_msg m;
    memset(&m, 0, sizeof(m));
    m.n = -2;
    char e[96] = "";
    if (err)
        av_strerror(err, e, sizeof(e));
    snprintf(m.err, sizeof(m.err), "%s%s%s", what, err ? ": " : "", e);
    fprintf(stderr, "zcdec: %s\n", m.err);
    send_msg(&m, NULL, 0);
    exit(1);
}

static enum AVPixelFormat get_vaapi(AVCodecContext *ctx, const enum AVPixelFormat *f)
{
    for (; *f != AV_PIX_FMT_NONE; f++)
        if (*f == AV_PIX_FMT_VAAPI)
            return *f;
    fprintf(stderr, "zcdec: the decoder offers no VA-API format for this stream\n");
    return AV_PIX_FMT_NONE;
}

/* frames handed out and not yet acked, oldest first */
static AVFrame *held[64];
static int nheld = 0, inflight = 4;

static void wait_ack(void)
{
    char b;
    for (;;) {
        ssize_t r = recv(sock, &b, 1, 0);
        if (r == 1)
            break;
        if (r < 0 && errno == EINTR)
            continue;
        fprintf(stderr, "zcdec: the reader went away\n");
        exit(2);
    }
    av_frame_free(&held[0]);
    memmove(held, held + 1, sizeof(held[0]) * --nheld);
}

static void hand_out(AVFrame *f, int64_t n)
{
    AVFrame *d = av_frame_alloc();
    if (!d)
        fail("out of memory", 0);
    d->format = AV_PIX_FMT_DRM_PRIME;
    int r = av_hwframe_map(d, f, AV_HWFRAME_MAP_READ);   /* vaExportSurfaceHandle, after the decode completed */
    if (r < 0)
        fail("mapping a VA surface to DRM PRIME failed", r);
    const AVDRMFrameDescriptor *desc = (const AVDRMFrameDescriptor *)d->data[0];
    AVHWFramesContext *fc = (AVHWFramesContext *)f->hw_frames_ctx->data;
    const AVPixFmtDescriptor *pd = av_pix_fmt_desc_get(fc->sw_format);
    struct zc_msg m;
    memset(&m, 0, sizeof(m));
    m.n = (int32_t)n;
    m.width = f->width;
    m.height = f->height;
    m.sw_format = fc->sw_format == AV_PIX_FMT_P010 ? 1 : fc->sw_format == AV_PIX_FMT_NV12 ? 2 : 0;
    m.bits = pd ? pd->comp[0].depth : 0;
    m.pts = f->pts;
    if (desc->nb_objects > ZC_MAX_OBJ || desc->nb_layers > ZC_MAX_LAYERS)
        fail("DRM PRIME descriptor too large", 0);
    m.nb_objects = desc->nb_objects;
    int fds[ZC_MAX_OBJ];
    for (int i = 0; i < desc->nb_objects; i++) {
        fds[i] = desc->objects[i].fd;
        m.size[i] = desc->objects[i].size;
        m.modifier[i] = desc->objects[i].format_modifier;
    }
    m.nb_layers = desc->nb_layers;
    for (int i = 0; i < desc->nb_layers; i++) {
        m.layer[i].format = desc->layers[i].format;
        m.layer[i].nb_planes = desc->layers[i].nb_planes;
        for (int p = 0; p < desc->layers[i].nb_planes && p < 4; p++) {
            m.layer[i].object[p] = desc->layers[i].planes[p].object_index;
            m.layer[i].offset[p] = desc->layers[i].planes[p].offset;
            m.layer[i].pitch[p] = desc->layers[i].planes[p].pitch;
        }
    }
    if (send_msg(&m, fds, desc->nb_objects) < 0) {
        fprintf(stderr, "zcdec: sending frame %lld failed: %s\n", (long long)n, strerror(errno));
        exit(2);
    }
    /* the mapping (and with it the exported fds and the surface) lives until the ack */
    av_frame_unref(f);
    held[nheld++] = d;
    while (nheld >= inflight)
        wait_ack();
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: zcdec SRC SOCKFD STEP [A:B ...]\n");
        return 64;
    }
    sock = atoi(argv[2]);
    step = atoi(argv[3]);
    for (int i = 4; i < argc && nranges < ZC_MAX_RANGES; i++) {
        long long a, b;
        if (sscanf(argv[i], "%lld:%lld", &a, &b) == 2) {
            ra[nranges] = a;
            rb[nranges++] = b;
        }
    }
    const char *e = getenv("ZC_INFLIGHT");
    if (e && atoi(e) > 0 && atoi(e) < 60)
        inflight = atoi(e);
    const char *node = getenv("RENDER_NODE") ? getenv("RENDER_NODE") : "/dev/dri/renderD128";

    av_log_set_level(AV_LOG_ERROR);
    AVFormatContext *fmt = NULL;
    int r = avformat_open_input(&fmt, argv[1], NULL, NULL);
    if (r < 0)
        fail("opening the source failed", r);
    if ((r = avformat_find_stream_info(fmt, NULL)) < 0)
        fail("reading the stream info failed", r);
    int vs = -1;
    for (unsigned i = 0; i < fmt->nb_streams; i++)
        if (fmt->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO &&
            !(fmt->streams[i]->disposition & AV_DISPOSITION_ATTACHED_PIC)) {
            vs = (int)i;           /* the first video stream, as -map 0:v:0 (attached pictures are not 0:v:0's) */
            break;
        }
    if (vs < 0)
        fail("no video stream", 0);
    for (unsigned i = 0; i < fmt->nb_streams; i++)
        fmt->streams[i]->discard = (int)i == vs ? AVDISCARD_DEFAULT : AVDISCARD_ALL;
    const AVCodec *codec = avcodec_find_decoder(fmt->streams[vs]->codecpar->codec_id);
    if (!codec)
        fail("no decoder for this codec in zcdec", 0);
    AVCodecContext *ctx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(ctx, fmt->streams[vs]->codecpar);
    ctx->pkt_timebase = fmt->streams[vs]->time_base;
    AVBufferRef *dev = NULL;
    if ((r = av_hwdevice_ctx_create(&dev, AV_HWDEVICE_TYPE_VAAPI, node, NULL, 0)) < 0)
        fail("creating the VA-API device failed", r);
    ctx->hw_device_ctx = av_buffer_ref(dev);
    ctx->get_format = get_vaapi;
    ctx->extra_hw_frames = inflight + 2;   /* the surfaces out with the reader are not the decoder's to reuse */
    ctx->thread_count = 1;
    if ((r = avcodec_open2(ctx, codec, NULL)) < 0)
        fail("opening the decoder failed", r);

    AVPacket *pkt = av_packet_alloc();
    AVFrame *f = av_frame_alloc();
    int64_t n = 0;
    int eof = 0;
    while (!eof) {
        r = av_read_frame(fmt, pkt);
        if (r == AVERROR_EOF) {
            eof = 1;
            r = avcodec_send_packet(ctx, NULL);
        } else if (r < 0) {
            fail("reading the source failed", r);
        } else {
            if (pkt->stream_index != vs) {
                av_packet_unref(pkt);
                continue;
            }
            r = avcodec_send_packet(ctx, pkt);
            av_packet_unref(pkt);
            if (r == AVERROR_INVALIDDATA)   /* ffmpeg's CLI logs it and goes on; so do we */
                r = 0;
        }
        if (r < 0 && r != AVERROR(EAGAIN) && r != AVERROR_EOF)
            fail("decoding failed", r);
        for (;;) {
            r = avcodec_receive_frame(ctx, f);
            if (r == AVERROR(EAGAIN) || r == AVERROR_EOF)
                break;
            if (r < 0)
                fail("decoding failed", r);
            if (f->format != AV_PIX_FMT_VAAPI)
                fail("the decoder did not return a VA-API surface", 0);
            if (kept(n))
                hand_out(f, n);
            else
                av_frame_unref(f);
            n++;
        }
    }
    /* the end first, then the acks still owed: the reader acks what it holds once it sees the end */
    struct zc_msg m;
    memset(&m, 0, sizeof(m));
    m.n = -1;
    m.pts = n;              /* frames decoded, for the reader's check */
    send_msg(&m, NULL, 0);
    while (nheld > 0)
        wait_ack();
    av_frame_free(&f);
    av_packet_free(&pkt);
    avcodec_free_context(&ctx);
    av_buffer_unref(&dev);
    avformat_close_input(&fmt);
    return 0;
}
