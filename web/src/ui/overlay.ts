import type { ObjectDetection } from "../types/messages";

export class OverlayRenderer {
  private readonly canvas: HTMLCanvasElement;
  private readonly context: CanvasRenderingContext2D;

  constructor(canvas: HTMLCanvasElement) {
    const context = canvas.getContext("2d");
    if (!context) {
      throw new Error("2D canvas context unavailable");
    }
    this.canvas = canvas;
    this.context = context;
  }

  syncWithVideo(video: HTMLVideoElement): void {
    const width = video.videoWidth || video.clientWidth;
    const height = video.videoHeight || video.clientHeight;
    if (width <= 0 || height <= 0) {
      return;
    }
    this.canvas.width = width;
    this.canvas.height = height;
  }

  clear(): void {
    this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  drawBoxes(objects: ObjectDetection[]): void {
    this.clear();
    this.context.lineWidth = 2;
    this.context.strokeStyle = "#f4a261";
    this.context.fillStyle = "rgba(15, 23, 42, 0.85)";
    this.context.font = "14px ui-monospace, SFMono-Regular, Menlo, monospace";

    for (const object of objects) {
      if (!object.bbox) {
        continue;
      }

      const [x, y, w, h] = object.bbox;
      const px = x * this.canvas.width;
      const py = y * this.canvas.height;
      const pw = w * this.canvas.width;
      const ph = h * this.canvas.height;

      this.context.strokeRect(px, py, pw, ph);
      const label = `${object.label} ${(object.confidence * 100).toFixed(0)}%`;
      this.context.fillRect(px, Math.max(0, py - 20), this.context.measureText(label).width + 12, 20);
      this.context.fillStyle = "#e9f1f7";
      this.context.fillText(label, px + 6, Math.max(14, py - 6));
      this.context.fillStyle = "rgba(15, 23, 42, 0.85)";
    }
  }
}
