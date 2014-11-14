package edu.stanford.nlp.kbp.slotfilling.ir.query;

import edu.stanford.nlp.util.Timing;

/**
 *
 * Statistics about how the query is going
 *
 * @author Angel Chang
 */
public class QueryStats {
  public int lastQueryHits;
  public int lastQueryTotalHits;
  public long totalElapsedMs;
  public long lastQueryElapsedMs;
  public Timing timing = new Timing();
}
