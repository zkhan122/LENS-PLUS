export class UiLogger {
  private readonly target: HTMLUListElement;
  private readonly maxEntries: number;

  constructor(target: HTMLUListElement, maxEntries = 30) {
    this.target = target;
    this.maxEntries = maxEntries;
  }

  log(message: string): void {
    const entry = document.createElement("li");
    const now = new Date().toLocaleTimeString();
    entry.textContent = `[${now}] ${message}`;

    this.target.prepend(entry);
    while (this.target.children.length > this.maxEntries) {
      this.target.removeChild(this.target.lastElementChild as Node);
    }
  }
}
