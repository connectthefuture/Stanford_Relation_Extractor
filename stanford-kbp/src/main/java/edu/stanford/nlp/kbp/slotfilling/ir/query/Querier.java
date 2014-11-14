package edu.stanford.nlp.kbp.slotfilling.ir.query;

import edu.stanford.nlp.kbp.common.KBPEntity;
import edu.stanford.nlp.kbp.common.NERTag;
import edu.stanford.nlp.kbp.common.Maybe;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IterableIterator;
import edu.stanford.nlp.util.Pair;

import java.util.Set;
import java.util.stream.Stream;

/**
 * An abstract specification of a querier, as a function from a query to sentences.
 *
 * Note that not all queriers will return coherent documents (e.g., web snippets)
 *
 * @author Gabor Angeli
 */
public interface Querier {
  /**
   * Query a collection of sentences based on a specification of things to query for.
   * Note that many terms could be undefined -- these are denoted with {@link edu.stanford.nlp.kbp.common.Maybe}s.
   * @param entity The query entity.
   * @param slotValue The slot value to query for
   * @param maxDocuments The maximum number of documents to query.
   * @return An iterable of sentences, along with their scores (e.g., Lucene scores). Often, these documents are lazily loaded.
   */
  public IterableIterator<Pair<CoreMap, Double>> querySentences(KBPEntity entity,
                                                                Maybe<KBPEntity> slotValue,
                                                                Maybe<String> relationName,
                                                                Set<String> docidsToForce,
                                                                Maybe<Integer> maxDocuments);

  public Stream<Annotation> slurp(int maxDocuments);

  /** Perform any close actions which may be relevant. This includes closing Lucene, file handles, etc. */
  public void close();
}
