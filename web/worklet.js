// Gapless playback AudioWorklet: a queue of Float32 sentence buffers played back-to-back.
// Messages in : {type:'push', buf:Float32Array, id, turn} | {type:'flush'}
// Messages out: {type:'started', id, turn} | {type:'idle'} | {type:'flushed', id, fraction}
class PlayerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.cur = null;
    this.pos = 0;
    this.lastId = -1;      // last sentence that started playing
    this.active = false;   // true while anything is playing/queued
    this.port.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'push') {
        this.queue.push({ buf: m.buf, id: m.id, turn: m.turn });
      } else if (m.type === 'flush') {
        const fraction = this.cur ? this.pos / this.cur.buf.length : 1;
        const id = this.lastId;
        this.queue = [];
        this.cur = null;
        this.pos = 0;
        this.lastId = -1;
        this.active = false;
        this.port.postMessage({ type: 'flushed', id, fraction });
      }
    };
  }

  process(_inputs, outputs) {
    const out = outputs[0][0];
    let i = 0;
    while (i < out.length) {
      if (!this.cur) {
        this.cur = this.queue.shift() || null;
        this.pos = 0;
        if (!this.cur) break;
        this.lastId = this.cur.id;
        this.active = true;
        this.port.postMessage({ type: 'started', id: this.cur.id, turn: this.cur.turn });
      }
      const n = Math.min(out.length - i, this.cur.buf.length - this.pos);
      out.set(this.cur.buf.subarray(this.pos, this.pos + n), i);
      this.pos += n;
      i += n;
      if (this.pos >= this.cur.buf.length) this.cur = null;
    }
    if (i < out.length) out.fill(0, i);
    if (this.active && !this.cur && this.queue.length === 0) {
      this.active = false;
      this.port.postMessage({ type: 'idle' });
    }
    return true;
  }
}
registerProcessor('player', PlayerProcessor);
