package edu.stanford.nlp.kbp.slotfilling.ir.query;

import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.util.IterableIterator;
import edu.stanford.nlp.util.Pair;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.search.BooleanClause;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * A querier which tries increasingly loose queries until it has
 * enough results (as determined by a threshold)
 *
 * @author Gabor Angeli
 */
public class BackoffLuceneQuerier extends ParameterizedLuceneQuerier {

  public final LuceneQuerierParams[] backoffOrder;

  public BackoffLuceneQuerier(IndexReader reader, LuceneQuerierParams[] paramsWithSlotFill) {
    super(reader, paramsWithSlotFill[0]);
    this.backoffOrder = paramsWithSlotFill;
  }

  public BackoffLuceneQuerier(File reader, LuceneQuerierParams[] paramsWithSlotFill)
      throws IOException {
    super(reader, paramsWithSlotFill[0]);
    this.backoffOrder = paramsWithSlotFill;
  }

  public BackoffLuceneQuerier(IndexReader reader) {
    super(reader, defaultBackoff[0]);
    this.backoffOrder = defaultBackoff;
  }

  public BackoffLuceneQuerier(File reader) throws IOException {
    super(reader, defaultBackoff[0]);
    this.backoffOrder = defaultBackoff;
  }

  @Override
  protected IterableIterator<Pair<Integer, Double>> queryImplementation(String entityName, Maybe<NERTag> entityType, Maybe<String> relation, Maybe<String> slotValue, Maybe<NERTag> slotValueType, Maybe<Integer> maxDocuments) throws IOException {
    return queryImplementation(backoffOrder, null, entityName, entityType, relation, slotValue, slotValueType, maxDocuments);
  }

  protected IterableIterator<Pair<Integer, Double>> queryImplementation(LuceneQuerierParams[] backoff,
                                                                        QueryStats queryStats,
                                                                        String entityName, Maybe<NERTag> entityType, Maybe<String> relation, Maybe<String> slotValue, Maybe<NERTag> slotValueType, Maybe<Integer> maxDocuments) throws IOException {
    // Overhead
    if (!maxDocuments.isDefined()) { throw new IllegalArgumentException("Cannot run backoff querier without max documents defined!"); }
    Set<Integer> seenDocuments = new HashSet<Integer>();
    // Run Queries
    List<Pair<Integer, Double>> responses = new ArrayList<Pair<Integer, Double>>(maxDocuments.get());
    OUT: for (int i = 0; i < backoff.length; ++i) {
      LuceneQuerierParams param = backoff[i];
      logger.log("backoff: got " + responses.size() + "/" + maxDocuments.get() + " so far");
      // Run query
      IterableIterator<Pair<Integer, Double>> candidateResponse;
      try {
        candidateResponse = super.queryImplementation(param, queryStats, entityName, entityType, relation, slotValue, slotValueType, maxDocuments);
      } catch (OutOfMemoryError e) {
        // If we out of memory, backoff gracefully
        logger.warn(e);
        continue;
      }
      // Register responses
      for (Pair<Integer, Double> response : candidateResponse) { // for each response from this query
        if (seenDocuments.contains(response.first)) {
          continue;
        }
        seenDocuments.add(response.first);
        responses.add(response);
        if (responses.size() >= maxDocuments.get()) {
          break OUT;
        }
      }
      if (responses.size() == 0 && i < backoff.length / 2 && backoff.length > (i + Props.INDEX_LUCENE_SKIPPINGBACKOFF + 2)) {
        i += Props.INDEX_LUCENE_SKIPPINGBACKOFF; // "No no no, go past this. Past this part." http://www.youtube.com/watch?v=qzO4BSTnkgg#t=0m30s
      }
    }
    logger.log("backoff: got " + responses.size() + "/" + maxDocuments.get() + " total" +
            ((queryStats != null)? (" in " + queryStats.totalElapsedMs + " msecs"):""));
    // Return
    return new IterableIterator<Pair<Integer, Double>>(responses.iterator());
  }

  public final static LuceneQuerierParams[] defaultBackoff = new LuceneQuerierParams[]{
      LuceneQuerierParams.strict(),
      LuceneQuerierParams.strict().withCaseSensitive(false),
      LuceneQuerierParams.strict().withCaseSensitive(false).withPhraseSemantics(LuceneQuerierParams.PhraseSemantics.SPAN_ORDERED),
      LuceneQuerierParams.strict().withCaseSensitive(false).withPhraseSemantics(LuceneQuerierParams.PhraseSemantics.SPAN_UNORDERED).withSlop(2),
      LuceneQuerierParams.strict().withCaseSensitive(false).withPhraseSemantics(LuceneQuerierParams.PhraseSemantics.SPAN_UNORDERED).withSlop(2).withRelation(false),
      LuceneQuerierParams.strict().withCaseSensitive(false).withPhraseSemantics(LuceneQuerierParams.PhraseSemantics.SPAN_UNORDERED).withSlop(2).withRelation(false).withSlotFillType(false),
      LuceneQuerierParams.strict().withCaseSensitive(false).withPhraseSemantics(LuceneQuerierParams.PhraseSemantics.UNIGRAMS_MUST),
      LuceneQuerierParams.strict().withCaseSensitive(false).withPhraseSemantics(LuceneQuerierParams.PhraseSemantics.UNIGRAMS_SHOULD),
      LuceneQuerierParams.strict().withCaseSensitive(false).withConjunctionMode(BooleanClause.Occur.SHOULD),
      LuceneQuerierParams.strict().withCaseSensitive(false).withPhraseSemantics(LuceneQuerierParams.PhraseSemantics.UNIGRAMS_MUST).withConjunctionMode(BooleanClause.Occur.SHOULD),
      LuceneQuerierParams.strict().withCaseSensitive(false).withPhraseSemantics(LuceneQuerierParams.PhraseSemantics.UNIGRAMS_SHOULD).withConjunctionMode(BooleanClause.Occur.SHOULD),
      LuceneQuerierParams.strict().withCaseSensitive(false).withPhraseSemantics(LuceneQuerierParams.PhraseSemantics.UNIGRAMS_SHOULD).withConjunctionMode(BooleanClause.Occur.SHOULD).withFuzzy(true),
      LuceneQuerierParams.strict().withCaseSensitive(false).withPhraseSemantics(LuceneQuerierParams.PhraseSemantics.UNIGRAMS_SHOULD).withConjunctionMode(BooleanClause.Occur.SHOULD).withFuzzy(true).withNERTag(false),
      LuceneQuerierParams.strict().withCaseSensitive(false).withPhraseSemantics(LuceneQuerierParams.PhraseSemantics.UNIGRAMS_SHOULD).withConjunctionMode(BooleanClause.Occur.SHOULD).withFuzzy(true).withNERTag(false).withRelation(false),
      LuceneQuerierParams.strict().withCaseSensitive(false).withPhraseSemantics(LuceneQuerierParams.PhraseSemantics.UNIGRAMS_SHOULD).withConjunctionMode(BooleanClause.Occur.SHOULD).withFuzzy(true).withNERTag(false).withRelation(false).withSlotFillType(false),
      LuceneQuerierParams.strict().withCaseSensitive(false).withPhraseSemantics(LuceneQuerierParams.PhraseSemantics.UNIGRAMS_SHOULD).withConjunctionMode(BooleanClause.Occur.SHOULD).withFuzzy(true).withNERTag(false).withRelation(false).withSlotFill(false).withSlotFillType(false)
  };

  public final static LuceneQuerierParams[] fastBackoff = new LuceneQuerierParams[]{
//      LuceneQuerierParams.strict().withCaseSensitive(false),
      LuceneQuerierParams.strict().withCaseSensitive(false).withPhraseSemantics(LuceneQuerierParams.PhraseSemantics.SPAN_ORDERED),
//      LuceneQuerierParams.strict().withCaseSensitive(false).withPhraseSemantics(LuceneQuerierParams.PhraseSemantics.SPAN_UNORDERED).withSlop(2).withRelation(false).withSlotFillType(false),
//      LuceneQuerierParams.strict().withCaseSensitive(false).withPhraseSemantics(LuceneQuerierParams.PhraseSemantics.UNIGRAMS_SHOULD).withConjunctionMode(BooleanClause.Occur.SHOULD),
//      LuceneQuerierParams.strict().withCaseSensitive(false).withPhraseSemantics(LuceneQuerierParams.PhraseSemantics.UNIGRAMS_SHOULD).withConjunctionMode(BooleanClause.Occur.SHOULD).withFuzzy(true).withNERTag(false),
//      LuceneQuerierParams.strict().withCaseSensitive(false).withPhraseSemantics(LuceneQuerierParams.PhraseSemantics.UNIGRAMS_SHOULD).withConjunctionMode(BooleanClause.Occur.SHOULD).withFuzzy(true).withNERTag(false).withRelation(false),
      LuceneQuerierParams.strict().withCaseSensitive(false).withPhraseSemantics(LuceneQuerierParams.PhraseSemantics.UNIGRAMS_SHOULD).withConjunctionMode(BooleanClause.Occur.SHOULD).withFuzzy(true).withNERTag(false).withRelation(false).withSlotFill(false).withSlotFillType(false)
  };

//  TODO: Figure out better backoff order if we just have entity only (no slotfill)
//  public final static LuceneQuerierParams[] defaultBackoffEntityOnly = new LuceneQuerierParams[]{
//      LuceneQuerierParams.strict().withCaseSensitive(false),
//      LuceneQuerierParams.strict().withCaseSensitive(false).withPhraseSemantics(LuceneQuerierParams.PhraseSemantics.SPAN_ORDERED),
//      LuceneQuerierParams.strict().withCaseSensitive(false).withPhraseSemantics(LuceneQuerierParams.PhraseSemantics.SPAN_UNORDERED).withSlop(2),
//      LuceneQuerierParams.strict().withCaseSensitive(false).withPhraseSemantics(LuceneQuerierParams.PhraseSemantics.UNIGRAMS_MUST),
//      LuceneQuerierParams.strict().withCaseSensitive(false).withPhraseSemantics(LuceneQuerierParams.PhraseSemantics.UNIGRAMS_MUST).withConjunctionMode(BooleanClause.Occur.SHOULD),
//      LuceneQuerierParams.strict().withCaseSensitive(false).withPhraseSemantics(LuceneQuerierParams.PhraseSemantics.UNIGRAMS_MUST).withConjunctionMode(BooleanClause.Occur.SHOULD).withFuzzy(true),
//      LuceneQuerierParams.strict().withCaseSensitive(false).withPhraseSemantics(LuceneQuerierParams.PhraseSemantics.UNIGRAMS_MUST).withConjunctionMode(BooleanClause.Occur.SHOULD).withRelation(false),
//      LuceneQuerierParams.strict().withCaseSensitive(false).withPhraseSemantics(LuceneQuerierParams.PhraseSemantics.UNIGRAMS_SHOULD).withConjunctionMode(BooleanClause.Occur.SHOULD).withFuzzy(true).withRelation(false)
//  };

  public final static Pair<LuceneQuerierParams[], Integer> defaultBackOffWithStart = Pair.makePair(defaultBackoff, 0);
  public final static Pair<LuceneQuerierParams[], Integer> fastBackOffWithStart = Pair.makePair(fastBackoff, 0);


}
