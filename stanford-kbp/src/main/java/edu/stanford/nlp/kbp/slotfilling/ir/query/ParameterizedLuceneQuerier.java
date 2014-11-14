package edu.stanford.nlp.kbp.slotfilling.ir.query;

import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.kbp.slotfilling.ir.index.KBPField;
import edu.stanford.nlp.util.IterableIterator;
import edu.stanford.nlp.util.Pair;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.*;
import org.apache.lucene.store.FSDirectory;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * A general class for running a wide range of Lucene query types.
 * The behavior of the class is parameterized by the LuceneQueryParams class.
 *
 * @author Gabor Angeli
 */
public class ParameterizedLuceneQuerier extends LuceneQuerier {

  protected final LuceneQuerierParams params;

  public ParameterizedLuceneQuerier(IndexReader reader, LuceneQuerierParams params) {
    super(reader);
    this.params = params;
    this.searcher.setSimilarity(params.similarity);
  }

  public ParameterizedLuceneQuerier(File reader, LuceneQuerierParams params) throws IOException {
    this(DirectoryReader.open(FSDirectory.open(reader), Props.INDEX_TERMINDEXDIVISOR), params);
  }


  @Override
  protected IterableIterator<Pair<Integer, Double>> queryImplementation(String entityName, Maybe<NERTag> entityType,
                                                                        Maybe<String> relation,
                                                                        Maybe<String> slotValue, Maybe<NERTag> slotValueType,
                                                                        Maybe<Integer> maxDocuments) throws IOException {
    return queryImplementation(params, null, entityName, entityType, relation, slotValue, slotValueType, maxDocuments);
  }

  @SuppressWarnings("ConstantConditions")
  protected IterableIterator<Pair<Integer, Double>> queryImplementation(LuceneQuerierParams params,
                                                                        QueryStats queryStats,
                                                                        String entityName, Maybe<NERTag> entityType,
                                                                        Maybe<String> relation,
                                                                        Maybe<String> slotValue, Maybe<NERTag> slotValueType,
                                                                        Maybe<Integer> maxDocuments) throws IOException {
    long startTime = 0;
    if (queryStats != null) {
      startTime = queryStats.timing.report();
    }
    List<Query> andClauses = new LinkedList<>();

    // -- Query Components
    // Entity query
    Query entityQuery = qrewrite(entityName, params);
    //andClauses.add( entityQuery );

    // Slot fill query
    Query slotValueQuery = null;
    if (params.querySlotFill && slotValue.isDefined()) {
      slotValueQuery = qrewrite(slotValue.get(), params);
      andClauses.add( slotValueQuery );
    }

    // If relation is defined (looking for provenance)
    // or we don't know the slot fill (looking for anything)
    // do we want to do relation specific query such as query for keyword and matching slotfill
    // Don't want to do this for training (when querying just for entity + slot)
    // maybe some of this logic should be controlled more by the caller...
    boolean doRelationSpecificQuery = relation.isDefined() || !slotValue.isDefined();

    // Relation query (e.g., keywords)
    // TODO: should we add keywords in even if slotValue is defined?
    Query keywordsQuery = null;
    if (doRelationSpecificQuery && params.queryRelation && Keywords.get().isDefined) {
      if (relation.isDefined()) {
        String relName = relation.get();
        Maybe<RelationType> relType = RelationType.fromString(relName);
        if (relType.isDefined()) {
          Collection<String> relKeywords = Keywords.get().getRelationKeywords(relType.get());
          if (relKeywords != null && !relKeywords.isEmpty()) {
            keywordsQuery = getKeywordsQuery(relKeywords);
          }
        } else {
          logger.warn("Cannot found relation type for " + relName);
        }
      } else if (entityType.isDefined()) {
        // TODO: query for a nice variation of relations
        // For now, just get all the relations for the entity type
        Collection<String> relKeywords = Keywords.get().getRelationKeywords(entityType.get());
        if (relKeywords != null && !relKeywords.isEmpty()) {
          keywordsQuery = getKeywordsQuery(relKeywords);
        }
      }
    }
    if (keywordsQuery != null) {
      andClauses.add(keywordsQuery);
    }

    // Entity type query
//    if (params.queryNERTag) {
//      // TODO(gabor) query the entity type
//    }

    // Slot fill type query
    // only query with slot fill type if slot fill not defined
    Query slotValueTypeQuery = null;
    if (doRelationSpecificQuery && params.querySlotFillType && supportsFillTypesQuery()) {
      if (slotValueType.isDefined()) {
        slotValueTypeQuery = getFillTypeQuery(slotValueType.get());
      } else if (relation.isDefined()) {
        // Get query slot fill types from relation
        String relName = relation.get();
        Maybe<RelationType> relType = RelationType.fromString(relName);
        if (relType.isDefined()) {
          slotValueTypeQuery = getFillTypeQuery(relType.get());
        } else {
          logger.warn("Cannot found relation type for " + relName);
        }
      }
    }
    if (slotValueTypeQuery != null) {
      andClauses.add(slotValueTypeQuery);
    }

    if (keywordsQuery != null) {
      // Have boost weights for entity/slotfill
      if (entityQuery != null && params.entityBoost > 0) entityQuery.setBoost(params.entityBoost);
      if (slotValueQuery != null && params.slotfillBoost > 0) slotValueQuery.setBoost(params.slotfillBoost);
    }

    // -- Combine Query
    BooleanQuery query = new BooleanQuery();
    // Reasonable to say we must always have entity?
    query.add(entityQuery, BooleanClause.Occur.MUST);
    for (Query clause : andClauses) {
      query.add(clause, params.conjunctionMode);
    }


    // -- Collect Results
    TopDocs docs = queryWithTimeout(query, maxDocuments, Props.INDEX_LUCENE_TIMEOUTMS);
    if (queryStats != null) {
      queryStats.lastQueryHits = docs.scoreDocs.length;
      queryStats.lastQueryTotalHits = docs.totalHits;
      queryStats.lastQueryElapsedMs = queryStats.timing.report() - startTime;
      queryStats.totalElapsedMs += queryStats.lastQueryElapsedMs;
    }
    logger.log("query: " + query + " got: " + docs.scoreDocs.length + "/" + docs.totalHits);
    return asIterator(docs);
  }

  // Utility functions for querying different fields
  private boolean supportsFillTypesQuery() {
    // we check the text annotated field because we don't have the TEXT_NER field stored..., just indexed
    return knownFields.contains(KBPField.TEXT_ANNOTATED.fieldName());
  }

  private Query getFillTypeQuery(RelationType relation) {
    Set<NERTag> fillTypes = relation.validNamedEntityLabels;
    if (!fillTypes.isEmpty()) {
      return getFillTypesQuery(fillTypes);
    } else {
      logger.warn("No slot fill types for " + relation);
      return null;
    }
  }

  private Query getFillTypeQuery(NERTag slotValueType) {
    return new TermQuery( new Term(KBPField.TEXT_NER.fieldName(), slotValueType.name ) );
  }

  private Query getFillTypesQuery(Collection<NERTag> fillTypes) {
    BooleanQuery fillTypeQuery = new BooleanQuery();
    for (NERTag fillType: fillTypes) {
      Query q = new TermQuery( new Term(KBPField.TEXT_NER.fieldName(), fillType.name ) );
      fillTypeQuery.add(q, BooleanClause.Occur.SHOULD);
    }
    return fillTypeQuery;
  }

  private static final LuceneQuerierParams keywordsParams = LuceneQuerierParams.strict().withCaseSensitive(false);
  private Query getKeywordsQuery(Collection<String> words) {
    BooleanQuery q = new BooleanQuery();
    for (String w:words) {
      q.add( qrewrite(w, keywordsParams), BooleanClause.Occur.SHOULD );
    }
    return q;
  }

}
