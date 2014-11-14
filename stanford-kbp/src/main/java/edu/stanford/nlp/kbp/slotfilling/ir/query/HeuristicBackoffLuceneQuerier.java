package edu.stanford.nlp.kbp.slotfilling.ir.query;

import edu.stanford.nlp.kbp.common.NERTag;
import edu.stanford.nlp.kbp.common.Maybe;
import edu.stanford.nlp.util.IterableIterator;
import edu.stanford.nlp.util.Pair;
import org.apache.lucene.index.IndexReader;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * Similar to BackoffLuceneQuerier but has uses various heuristics
 *  1. Uses separate backoffs for different types of query input
 *  2. Starts in the middle of backoffs
 *
 * @author Angel Chang
 */
public class HeuristicBackoffLuceneQuerier extends BackoffLuceneQuerier {

  enum BackoffCondition { DEFAULT, ENTITY_ONLY, ENTITY_SLOT_FILL }

  // Based on whether the query includes certain components
  // uses a different backoff
  public EnumMap<BackoffCondition, Pair<LuceneQuerierParams[], Integer>> backoffs;

  public HeuristicBackoffLuceneQuerier(IndexReader reader,
                                       Pair<LuceneQuerierParams[], Integer> defaultBackoffOrder)
  {
    super(reader, defaultBackoffOrder.first);
    initExtraBackoffs(defaultBackoffOrder);
  }

  public HeuristicBackoffLuceneQuerier(File reader,
                                       Pair<LuceneQuerierParams[], Integer> defaultBackoffOrder) throws IOException {
    super(reader, defaultBackoffOrder.first);
    initExtraBackoffs(defaultBackoffOrder);
  }

  public HeuristicBackoffLuceneQuerier(IndexReader reader) {
    super(reader);
    initExtraBackoffs(defaultBackOffWithStart);
  }

  public HeuristicBackoffLuceneQuerier(File reader) throws IOException {
    super(reader);
    initExtraBackoffs(defaultBackOffWithStart);
  }

  // Initialize other backoffs from the default backoff
  private void initExtraBackoffs(Pair<LuceneQuerierParams[], Integer> defaultBackoffOrder) {
    this.backoffs = new EnumMap<BackoffCondition, Pair<LuceneQuerierParams[], Integer>>(BackoffCondition.class);
    this.backoffs.put(BackoffCondition.DEFAULT, defaultBackoffOrder);

    int defaultStartIndex = defaultBackoffOrder.second;

    // Entity slot fill backoff
    // Remove items that don't matter
    int i = 0;
    Set<LuceneQuerierParams> entitySlotFillParams = new LinkedHashSet<LuceneQuerierParams>();
    int entitySlotFillStartIndex = -1;
    for (LuceneQuerierParams params:defaultBackoffOrder.first) {
      // slot fill type and relation are not queried for if we have both entity and slot fill value
      // TODO: change this if relation/slotfilltype are also queried for
      entitySlotFillParams.add(params.withRelation(false).withSlotFillType(false));
      if (i == defaultStartIndex) {
        entitySlotFillStartIndex = entitySlotFillParams.size()-1;
      }
      i++;
    }
    this.backoffs.put(
             BackoffCondition.ENTITY_SLOT_FILL,
             Pair.makePair(entitySlotFillParams.toArray(new LuceneQuerierParams[entitySlotFillParams.size()]),
                           entitySlotFillStartIndex));

    // Entity only backoff
    i = 0;
    LinkedHashSet<LuceneQuerierParams> entityOnlyParams = new LinkedHashSet<LuceneQuerierParams>();
    int entityOnlyStartIndex = -1;
    for (LuceneQuerierParams params:defaultBackoffOrder.first) {
      // No slot fill anyways
      entityOnlyParams.add(params.withSlotFill(false));
      if (i == defaultStartIndex) {
        entityOnlyStartIndex = entityOnlyParams.size()-1;
      }
      i++;
    }
    this.backoffs.put(
            BackoffCondition.ENTITY_ONLY,
            Pair.makePair(entityOnlyParams.toArray(new LuceneQuerierParams[entityOnlyParams.size()]),
                    entityOnlyStartIndex));

    for (BackoffCondition cond:backoffs.keySet()) {
      logger.debug("backoff " + cond + ": " + backoffs.get(cond).first.length + ", start " + backoffs.get(cond).second);
    }
  }

  protected IterableIterator<Pair<Integer, Double>> queryImplementation(LuceneQuerierParams[] backoff,
                                                                        int startBackOffAt,
                                                                        QueryStats overallQueryStats,
                                                                        String entityName, Maybe<NERTag> entityType, Maybe<String> relation, Maybe<String> slotValue, Maybe<NERTag> slotValueType, Maybe<Integer> maxDocuments) throws IOException {
    // Overhead
    if (!maxDocuments.isDefined()) { throw new IllegalArgumentException("Cannot run backoff querier without max documents defined!"); }
    int maxDocs = maxDocuments.get();
    int acceptQueryThreshold = maxDocs*2;

    Pair< QueryStats,IterableIterator<Pair<Integer, Double>> >[] queryResults = new Pair[backoff.length];

    // Run Queries
    boolean doNormalBackoff = false;
    boolean done = false;
    Set<Integer> seenDocuments = new HashSet<Integer>();
    int index = (startBackOffAt >= 0)? startBackOffAt:0;
    List<Pair<Integer, Double>> responses = new ArrayList<Pair<Integer, Double>>(maxDocuments.get());
    while (!done && index < backoff.length) {
      try {
        QueryStats queryStats = null;
        IterableIterator<Pair<Integer, Double>> candidateResponse = null;
        if (queryResults[index] == null) {
          // query index!
          queryStats = new QueryStats();
          LuceneQuerierParams param = backoff[index];
          candidateResponse = queryImplementation(param, queryStats, entityName, entityType, relation, slotValue, slotValueType, maxDocuments);
          queryResults[index] = Pair.makePair(queryStats, candidateResponse);
          if (overallQueryStats != null) {
            overallQueryStats.lastQueryElapsedMs = queryStats.lastQueryElapsedMs;
            overallQueryStats.totalElapsedMs += queryStats.totalElapsedMs;
          }
        } else {
          queryStats = queryResults[index].first;
          candidateResponse = queryResults[index].second;
        }

        if (!doNormalBackoff) {
          // Not doing normal backoff - trying different queries until we get too little
          if (index > 0 && queryStats.lastQueryTotalHits > acceptQueryThreshold) {
            // Too many responses, try less
            index = index/2;
          } else {
            // Found a good point to start: continue with normal backoff now
            doNormalBackoff = true;
          }
        }

        // Normal backoff
        if (doNormalBackoff) {
          logger.log("heur backoff " + index + ": got " + responses.size() + "/" + maxDocuments.get() + " so far");
          for (Pair<Integer, Double> response : candidateResponse) { // for each response from this query
            if (seenDocuments.contains(response.first)) { continue; }
            seenDocuments.add(response.first);
            responses.add(response);
            if (responses.size() >= maxDocuments.get()) {
              done = true;
              break;
            }
          }
          index++;
        }
      } catch (OutOfMemoryError e) {
        index += 1;
        logger.warn(e);
        //noinspection UnnecessaryContinue
        continue;
      }
    }
    logger.log("heur backoff: got " + responses.size() + "/" + maxDocuments.get() + " total" +
      ((overallQueryStats != null)? (" in " + overallQueryStats.totalElapsedMs + " msecs"):""));

    // Return
    return new IterableIterator<Pair<Integer, Double>>(responses.iterator());
  }


  @Override
  protected IterableIterator<Pair<Integer, Double>> queryImplementation(String entityName, Maybe<NERTag> entityType, Maybe<String> relation, Maybe<String> slotValue, Maybe<NERTag> slotValueType, Maybe<Integer> maxDocuments) throws IOException {
    // TODO: can start in the middle of the backoff chain to get more results faster (loses some ordering)
    //   check queryStats to see if too many results (> some threshold) if too many, try normal backoff...
    Pair<LuceneQuerierParams[], Integer> selectedBackoff;
    if (slotValue.isDefined()) {
      // We have the slot fill defined
      if (relation.isDefined()) {
        // This is the query for provenance
        selectedBackoff = backoffs.get(BackoffCondition.DEFAULT);
      } else {
        // This is query for training
        selectedBackoff = backoffs.get(BackoffCondition.ENTITY_SLOT_FILL);
      }
    } else if (relation.isNothing() && slotValue.isNothing() && slotValueType.isNothing()) {
      // We only have the entity defined
      // This is the query used during test time to get sentences that we check to see if there is any relations
      selectedBackoff = backoffs.get(BackoffCondition.ENTITY_ONLY);
    } else {
      selectedBackoff = backoffs.get(BackoffCondition.DEFAULT);
    }
    QueryStats queryStats = new QueryStats();
    return queryImplementation(selectedBackoff.first, selectedBackoff.second, queryStats,
            entityName, entityType, relation, slotValue, slotValueType, maxDocuments);
  }

}
