package edu.stanford.nlp.kbp.common;

/**
 * Java locks are weird. This one is simple and works.
 * Like, you can unlock it from different threads.
 * Seriously, I hate you, Java locks. A lot.
 *
 * @author Gabor Angeli
 */
public class SimpleLock {

  private boolean isLocked = false;

  public synchronized void acquire() {
    while (this.isLocked) {
      try {
        this.wait();
      } catch (InterruptedException ignored) { }
    }
    this.isLocked = true;
  }

  public synchronized void release() {
    this.isLocked = false;
    this.notifyAll();
  }
}
