package edu.stanford.nlp.kbp.slotfilling.ir;

import edu.stanford.nlp.kbp.common.KBPEntity;
import edu.stanford.nlp.kbp.common.Maybe;
import edu.stanford.nlp.kbp.slotfilling.ir.query.LuceneQuerierParams;
import edu.stanford.nlp.kbp.slotfilling.ir.query.Querier;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IterableIterator;
import edu.stanford.nlp.util.Pair;

import java.io.File;
import java.util.Collections;
import java.util.Set;
import java.util.stream.Stream;

/**
 * TODO(gabor) JavaDoc
 *
 * @author Gabor Angeli
 */
public class NOOPLuceneQuerier implements Querier {
  public NOOPLuceneQuerier(File file, LuceneQuerierParams base) {
  }

  @SuppressWarnings("unchecked")
  @Override
  public IterableIterator<Pair<CoreMap, Double>> querySentences(KBPEntity entity, Maybe<KBPEntity> slotValue, Maybe<String> relationName, Set<String> docidsToForce, Maybe<Integer> maxDocuments) {
    return new IterableIterator<Pair<CoreMap, Double>>(Collections.EMPTY_LIST.iterator());
  }

  @Override
  public Stream<Annotation> slurp(int maxDocuments) {
    return Stream.empty();
  }

  @Override
  public void close() {
    // do nothing
  }
}
