export class FileSource {
  private hiddenVideo: HTMLVideoElement | null = null;
  private objectUrl: string | null = null;
  private fallbackCanvas: HTMLCanvasElement | null = null;
  private fallbackRafId: number | null = null;

  async start(file: File): Promise<MediaStream> {
    this.cleanup();

    this.hiddenVideo = document.createElement("video");
    this.hiddenVideo.muted = true;
    this.hiddenVideo.loop = true;
    this.hiddenVideo.playsInline = true;

    this.objectUrl = URL.createObjectURL(file);
    this.hiddenVideo.src = this.objectUrl;

    await this.hiddenVideo.play();

    const stream = this.captureFromVideo(this.hiddenVideo);
    if (stream.getVideoTracks().length === 0) {
      throw new Error("Could not capture video track from file");
    }
    return stream;
  }

  private captureFromVideo(video: HTMLVideoElement): MediaStream {
    const captureCapable = video as HTMLVideoElement & {
      captureStream?: () => MediaStream;
      webkitCaptureStream?: () => MediaStream;
    };

    if (typeof captureCapable.captureStream === "function") {
      return captureCapable.captureStream();
    }

    if (typeof captureCapable.webkitCaptureStream === "function") {
      return captureCapable.webkitCaptureStream();
    }

    this.fallbackCanvas = document.createElement("canvas");
    this.fallbackCanvas.width = video.videoWidth || 1280;
    this.fallbackCanvas.height = video.videoHeight || 720;
    const context = this.fallbackCanvas.getContext("2d");
    if (!context) {
      throw new Error("Could not initialize fallback canvas capture");
    }

    const renderFrame = (): void => {
      if (!this.hiddenVideo || !this.fallbackCanvas) {
        return;
      }
      context.drawImage(this.hiddenVideo, 0, 0, this.fallbackCanvas.width, this.fallbackCanvas.height);
      this.fallbackRafId = window.requestAnimationFrame(renderFrame);
    };
    renderFrame();

    const canvasStream = this.fallbackCanvas.captureStream(30);
    if (canvasStream.getVideoTracks().length === 0) {
      throw new Error("Fallback canvas stream does not contain video tracks");
    }
    return canvasStream;
  }

  cleanup(): void {
    if (this.fallbackRafId !== null) {
      window.cancelAnimationFrame(this.fallbackRafId);
      this.fallbackRafId = null;
    }
    this.fallbackCanvas = null;

    if (this.hiddenVideo) {
      this.hiddenVideo.pause();
      this.hiddenVideo.src = "";
      this.hiddenVideo = null;
    }
    if (this.objectUrl) {
      URL.revokeObjectURL(this.objectUrl);
      this.objectUrl = null;
    }
  }
}
