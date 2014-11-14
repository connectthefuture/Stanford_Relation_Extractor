package edu.stanford.nlp.kbp.slotfilling.ir;

import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.kbp.common.CollectionUtils;
import edu.stanford.nlp.kbp.slotfilling.evaluate.GoldResponseSet;
import edu.stanford.nlp.kbp.slotfilling.ir.index.KryoAnnotationSerializer;
import edu.stanford.nlp.kbp.slotfilling.ir.query.*;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.util.*;
import edu.stanford.nlp.io.IOUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.lang.ref.SoftReference;
import java.sql.Connection;
import java.sql.SQLException;
import java.util.*;
import java.util.stream.Stream;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * Quick interface to query documents (annotated and featurized).
 */
public class
  StandardIR extends KBPIR {

  /**
   * An object representing an IR query.
   * This class should not change often, or else it will invalidate the cache!
   */
  public static final class QueryBundle implements Serializable {
    private static final long serialVersionUID = 1L;

    public final Maybe<List<Integer>> docIds;
    public final Maybe<String> entity1;
    public final Maybe<NERTag> entity1Type;
    public final Maybe<String> entity2;
    public final Maybe<String> relation;
    public final int numSentences;

    public QueryBundle(Maybe<List<Integer>> docIds, Maybe<String> entity1, Maybe<NERTag> entity1Type,
                       Maybe<String> entity2, Maybe<String> relation, int numSentences) {
      this.docIds = docIds;
      this.entity1 = entity1;
      this.entity1Type = entity1Type;
      this.entity2 = entity2;
      this.relation = relation;
      this.numSentences = numSentences;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (!(o instanceof QueryBundle)) return false;
      QueryBundle that = (QueryBundle) o;
      return numSentences == that.numSentences && docIds.equals(that.docIds) && entity1.equals(that.entity1) && entity1Type.equals(that.entity1Type) && entity2.equals(that.entity2) && relation.equals(that.relation);
    }

    @Override
    public int hashCode() {
      int result = docIds.hashCode();
      result = 31 * result + entity1.hashCode();
      result = 31 * result + entity1Type.hashCode();
      result = 31 * result + entity2.hashCode();
      result = 31 * result + relation.hashCode();
      result = 31 * result + numSentences;
      return result;
    }

    @Override
    public String toString() {
      return "QueryBundle{" +
          "docIds=" + StringUtils.join(docIds.getOrElse(new LinkedList<Integer>()), ",") +
          ", entity1=" + entity1.getOrElse("---") +
          ", entity1Type=" + entity1Type.getOrElse(null) +
          ", entity2=" + entity2.getOrElse("---") +
          ", relation=" + relation.getOrElse("---") +
          ", numSentences=" + numSentences +
          '}';
    }
  }


  //
  // Variables
  //
  private final Querier[] backends;
  public final LuceneQuerier officialIndex;
  private final AnnotationPipeline reannotatePipeline;
  public final AnnotationSerializer serializer = new KryoAnnotationSerializer();

  /**
   * A cache to avoid having to lookup a document every single time from Lucene in case we're
   * looking up the same document again and again.
   * This will keep up to 10k documents in the cache, freeing them if memory becomes an issue.
   *
   * @see StandardIR#fetchDocument(String, boolean)
   */
  private final Map<String, SoftReference<Annotation>> cachedDocumentLookup = new HashMap<>();

  //
  // Constructor
  //
  public StandardIR(Properties props, Maybe<GoldResponseSet> goldProvenances) {
    // Set reannotation pipeline (normally not needed)
    if (Props.INDEX_REANNOTATE != null && !Props.INDEX_REANNOTATE.isEmpty()) {
      Properties pipelineProps;
      if (Props.INDEX_REANNOTATE.endsWith(".props") || Props.INDEX_REANNOTATE.endsWith(".properties")) {
        // properties file
        try {
          BufferedReader br = IOUtils.getBufferedFileReader(Props.INDEX_REANNOTATE);
          pipelineProps = new Properties();
          pipelineProps.load(br);
          br.close();
        } catch (IOException ex) {
          throw new RuntimeException("Cannot initialize annotation pipeline for index: error reading from " + Props.INDEX_REANNOTATE, ex);
        }
      } else {
        // Just annotators
        pipelineProps = new Properties(props);
        pipelineProps.setProperty("annotators", Props.INDEX_REANNOTATE);
      }
      // Create new pipeline with our annotators
      // Don't enforce requirements since we are annotating on top
      //  of existing annotations
      logger.log("Setting index to reannotate with annotators: " + pipelineProps.get("annotators"));
      reannotatePipeline = new StanfordCoreNLP(pipelineProps, false);
    } else {
      reannotatePipeline = null;
    }

    // Initialize Index
    this.backends = constructBackends(goldProvenances);
    LuceneQuerier official = null;
    for (LuceneQuerier backend : luceneBackends()) {
      for (File indexDir : backend.indexDirectory) {
        try {
          if (indexDir.getCanonicalPath().equals(Props.INDEX_OFFICIAL.getCanonicalPath())) { official = backend; }
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      }
    }
    if (official == null) {
      if (Props.INDEX_MODE == Props.QueryMode.NOOP) {
        officialIndex = null;
      } else {
        throw new IllegalArgumentException("Official index was not found among index paths");
      }
    } else {
      officialIndex = official;
    }
    logger.log("official index is: " + officialIndex);
  }


  private Querier[] constructBackends(Maybe<GoldResponseSet> goldProvenances) {
    Querier[] backends = new Querier[0];
    try {
      if (goldProvenances.isDefined()) {
        backends = new Querier[1];
        backends[0] = new LuceneGoldQuerier(Props.INDEX_OFFICIAL, goldProvenances.get());
        logger.log("(gold) backend: " + backends[0]);
      } else {
        backends = new Querier[Props.INDEX_PATHS.length + (Props.INDEX_WEBSNIPPETS_DO ? 1 : 0)];
        // Add lucene queriers
        for (int i = 0; i < Props.INDEX_PATHS.length; ++i) {
          switch (Props.INDEX_MODE) {
            case NOOP:
              backends[i] = new NOOPLuceneQuerier(Props.INDEX_PATHS[i], LuceneQuerierParams.base());
              break;
            case DUMB:
              backends[i] = new ParameterizedLuceneQuerier(Props.INDEX_PATHS[i], LuceneQuerierParams.base());
              break;
            case REGULAR:
              backends[i] = new ParameterizedLuceneQuerier(Props.INDEX_PATHS[i], LuceneQuerierParams.base());
              break;
            case BACKOFF:
              backends[i] = new BackoffLuceneQuerier(Props.INDEX_PATHS[i], Props.INDEX_FASTBACKOFF ? BackoffLuceneQuerier.fastBackoff : BackoffLuceneQuerier.defaultBackoff);
              break;
            case HEURISTIC_BACKOFF:
              backends[i] = new HeuristicBackoffLuceneQuerier(Props.INDEX_PATHS[i], Props.INDEX_FASTBACKOFF ? BackoffLuceneQuerier.fastBackOffWithStart : BackoffLuceneQuerier.defaultBackOffWithStart);
              break;
            default:
          }
          if (backends[i] instanceof LuceneQuerier) {
            ((LuceneQuerier) backends[i]).setReannotatePipeline(reannotatePipeline);
          }
          logger.log("backend: " + backends[i]);
        }
        // Add webqueries
        if (Props.INDEX_WEBSNIPPETS_DO) {
          backends[Props.INDEX_PATHS.length] = new DirectFileQuerier(Props.INDEX_WEBSNIPPETS_DIR);
          logger.log("backend: " + backends[Props.INDEX_PATHS.length]);
        }
      }
    } catch (IOException e) {
      logger.fatal(e);
    }
    return backends;
  }

  private LuceneQuerier[] luceneBackends() {
    List<LuceneQuerier> lucenes = new ArrayList<>();
    for (Querier q : backends) {
      if (q instanceof  LuceneQuerier) {
        lucenes.add((LuceneQuerier) q);
      }
    }
    return lucenes.toArray(new LuceneQuerier[lucenes.size()]);
  }

  public StandardIR(Properties props, GoldResponseSet goldProvenances) {
    this(props, Maybe.Just(goldProvenances));
  }

  public StandardIR(Properties props) {
    this(props, Maybe.<GoldResponseSet>Nothing());
  }

  public void close() throws IOException {
    for (Querier backend : backends) {
      backend.close();
    }
  }

  private IterableIterator<Pair<CoreMap, Double>> queryImplementationSentence(
                                         final KBPEntity entity, final Maybe<KBPEntity> slotValue,
                                         final Maybe<String> relation,
                                         final Set<String> docidsToForce,
                                         final int maxDocuments, final boolean officialIndexOnly) {
    if (officialIndexOnly && Props.INDEX_MODE != Props.QueryMode.NOOP ) {
      return officialIndex.querySentences(entity, slotValue, relation, docidsToForce, Maybe.Just(maxDocuments));
    } else {
      return CollectionUtils.interleave(CollectionUtils.map(backends,
          in -> in.querySentences(entity, slotValue, relation, docidsToForce, Maybe.Just(maxDocuments))));
    }
  }

  private IterableIterator<Pair<Annotation, Double>> queryImplementationDocument(
      final KBPEntity entity, final Maybe<KBPEntity> slotValue,
      final Set<String> docidsToForce,
      final int maxDocuments, final boolean officialIndexOnly) {
    if (officialIndexOnly && Props.INDEX_MODE != Props.QueryMode.NOOP ) {
      return officialIndex.queryDocument(entity, slotValue, docidsToForce, Maybe.Just(maxDocuments));
    } else {
      return CollectionUtils.interleave(CollectionUtils.map(luceneBackends(),
          in -> in.queryDocument(entity, slotValue, docidsToForce, Maybe.Just(maxDocuments))));
    }
  }


  @Override
  protected <E extends CoreMap> List<E> queryCoreMaps(final String tableName, final Class<E> expectedOutput,
                                                      final KBPEntity entity, final Maybe<KBPEntity> slotValue,
                                                      final Maybe<String> relation, final Set<String> docidsToForce,
                                                      final int maxDocuments, final boolean officialIndexOnly) {
    startTrack("IR Query [" + entity.name + ", " + slotValue.orNull() + "]");
    final Pointer<List<E>> sentences = new Pointer<>();

    // Get table to read from
    final String table;
    if (Annotation.class.isAssignableFrom(expectedOutput)) {
      table = Props.DB_TABLE_DOCUMENT_CACHE;
    } else if (CoreMap.class.isAssignableFrom(expectedOutput)) {
      table = Props.DB_TABLE_SENTENCE_CACHE;
    } else {
      throw new IllegalArgumentException("Unknown query target (class): " + expectedOutput);
    }

    final boolean doCache;
    synchronized (Props.PROPERTY_CHANGE_LOCK) { doCache = Props.CACHE_SENTENCES_DO; }

    PostgresUtils.withKeyAnnotationTable(table, new PostgresUtils.KeyAnnotationCallback(this.serializer){ @SuppressWarnings("unchecked")
                                                                                                                                  @Override public void apply(Connection psql) throws SQLException {
      String key;
      if (slotValue.isDefined()) {
        key = keyToString(new QueryBundle(Maybe.<List<Integer>>Nothing(), Maybe.Just(entity.name), Maybe.Just(entity.type),
            Maybe.Just(slotValue.get().name), Maybe.<String>Nothing(), maxDocuments));
      } else {
        key = keyToString(new QueryBundle(Maybe.<List<Integer>>Nothing(), Maybe.Just(entity.name), Maybe.Just(entity.type),
            Maybe.<String>Nothing(), Maybe.<String>Nothing(), maxDocuments));
      }

      // Try Cache
      if (doCache && !Props.CACHE_SENTENCES_REDO) {
        for (List<Annotation> ann : get(psql, table, key)) {
          if (Annotation.class.isAssignableFrom(expectedOutput)) {
            sentences.set((List<E>) ann);  // case: retrieved documents
          } else if (expectedOutput.equals(CoreMap.class)) {
            assert ann.size() == 1;
            sentences.set((List<E>) ann.get(0).get(CoreAnnotations.SentencesAnnotation.class));  // case: retrieved sentences
          } else { throw new IllegalArgumentException("Unknown query target (class): " + expectedOutput); }
        }
      }

      // Run Query If Must
      if (!sentences.dereference().isDefined()) {
        // Run Query
        Object irResultObject;
        switch (table) {
          case Props.DB_TABLE_DOCUMENT_CACHE:
            irResultObject = queryImplementationDocument(entity, slotValue, docidsToForce, maxDocuments, officialIndexOnly);
            break;
          case Props.DB_TABLE_SENTENCE_CACHE:
            irResultObject = queryImplementationSentence(entity, slotValue, relation, docidsToForce, maxDocuments, officialIndexOnly);
            break;
          default:
            throw new IllegalArgumentException("Unknown query target (class): " + expectedOutput);
        }
        IterableIterator<Pair<E, Double>> irResults;
        if (irResultObject != null) {
          irResults =(IterableIterator<Pair<E, Double>>) irResultObject;
        } else { throw new IllegalStateException("A querier implementation returned null results!"); }

        // Process Query Result
        ArrayList<E> resultsAsList = new ArrayList<>();
        Set<String> sentencesSeen = new HashSet<>();
        for (Pair<E, Double> pair : irResults) {
          if (!sentencesSeen.contains(CoreMapUtils.sentenceToMinimalString(pair.first))) {
            resultsAsList.add(pair.first);
          }
          sentencesSeen.add(CoreMapUtils.sentenceToMinimalString(pair.first));
        }
        assert sentencesSeen.size() == resultsAsList.size();
        sentences.set(resultsAsList);

        // Cache
        if (doCache) {
          if (Annotation.class.isAssignableFrom(expectedOutput)) {
            put(psql, table, key, (List<Annotation>) resultsAsList);
          } else if (expectedOutput.equals(CoreMap.class)) {
            List<Annotation> toSave = Collections.singletonList(new Annotation(""));
            toSave.get(0).set(CoreAnnotations.SentencesAnnotation.class, (List<CoreMap>) resultsAsList);
            put(psql, table, key, toSave);
          } else { throw new IllegalArgumentException("Unknown query target (class): " + expectedOutput); }
          if (Props.KBP_EVALUATE && !psql.getAutoCommit()) { psql.commit(); }  // commit after every query -- slower, but can stop run in the middle
        }
      }
    }});
    endTrack("IR Query [" + entity.name + ", " + slotValue.orNull() + "]");
    return sentences.dereference().orCrash();
  }

  @Override
  protected List<String> queryDocIDs(final String entityName, final Maybe<NERTag> entityType, final Maybe<String> relation, final Maybe<String> slotValue, final Maybe<NERTag> slotValueType, final int maxDocuments, boolean officialIndexOnly) {
    // Run Query
    IterableIterator<Pair<String, Double>> irResults;
    if (officialIndexOnly && Props.INDEX_MODE != Props.QueryMode.NOOP ) {
      irResults = officialIndex.queryDocId(entityName, entityType, relation, slotValue, slotValueType, Maybe.Just(maxDocuments));
    } else {
      irResults = CollectionUtils.interleave(CollectionUtils.map(luceneBackends(),
          in -> in.queryDocId(entityName, entityType, relation, slotValue, slotValueType, Maybe.Just(maxDocuments))));
    }
    ArrayList<String> resultsAsList = new ArrayList<>();
    for (Pair<String, Double> pair : irResults) {
      resultsAsList.add(pair.first);
    }
    return resultsAsList;
  }

  /**
   * Queries all the indices to find the doc id, returning the first one
   * found.
   */
  public Annotation fetchDocument(String docId, boolean officialIndexOnly) throws IllegalArgumentException {
    // Check cache
    SoftReference<Annotation> ref = cachedDocumentLookup.get(docId);
    if (ref != null) {
      Annotation cachedValue = ref.get();
      if (cachedValue != null) {
        return cachedValue;
      }
    }
    try{
      Annotation result;
      // Run Query
      if (officialIndexOnly && Props.INDEX_MODE != Props.QueryMode.NOOP ) {
        result = officialIndex.fetchDocument(docId).orCrash("No such docid: " + docId);
      } else {
        Maybe<Annotation> doc = Maybe.Nothing();
        for( LuceneQuerier querier : luceneBackends() ) {
          doc = doc.orElse(querier.fetchDocument( docId ));
          if (doc.isDefined()) { break; }
        }
        result =  doc.orCrash("No such docid: " + docId);
      }
      // Do cache
      cachedDocumentLookup.put(docId, new SoftReference<>(result));
      // Make sure the cache size stays reasonable
      if (cachedDocumentLookup.size() > 10000) {
        // (remove stale references)
        Iterator<Map.Entry<String,SoftReference<Annotation>>> iter = cachedDocumentLookup.entrySet().iterator();
        while (iter.hasNext()) {
          if (iter.next().getValue().get() == null) { iter.remove(); }
        }
        // (make sure we're not storing too many things in the cache)
        iter = cachedDocumentLookup.entrySet().iterator();
        int originalSize = cachedDocumentLookup.size();
        for (int i = 0; i < originalSize - 5000; ++i) {
          iter.next();
          iter.remove();
        }
      }
      // Return
      return result;
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  /** {@inheritDoc} */
  @Override
  public int queryNumHits(Collection<String> terms) {
    int count = 0;
    for (Querier impl : this.backends) {
      if (impl instanceof LuceneQuerier) {
        count += ((LuceneQuerier) impl).queryHitCount(terms);
      }
    }
    return count;
  }

  /** {@inheritDoc} */
  @Override
  public Stream<Annotation> slurpDocuments(int maxDocuments) {
    if (Props.SHALLOWDIVE_OFFICIALONLY && Props.INDEX_MODE != Props.QueryMode.NOOP ) {
      return officialIndex.slurp(maxDocuments);
    } else {
      Stream s = Stream.empty();
      maxDocuments = maxDocuments / backends.length;
      for (Querier backend : backends) {
        if (backend instanceof LuceneQuerier) {
          s = Stream.concat(s, backend.slurp(maxDocuments));
        }
      }
      return s;
    }
  }

  /**
   * Ask the knowledge pair if it knows about any valid relations between given
   * entity pair.
   * @param pair The KBPair we are querying relations for
   */
  @Override
  public Set<String> getKnownRelationsForPair(KBPair pair) {
    KBPEntity entity = pair.getEntity();
    // Get all facts for the entity and search through them.
    if (knowledgeBase.data.isEmpty() && !Props.HACKS_DONTREADKB) {
      warn("getting relations without having loaded KB! Loading training KB by default");
      this.trainingTriples();
    }
    Collection<KBPSlotFill> triples = knowledgeBase.get(entity).getOrElse(new HashSet<KBPSlotFill>());

    Set<String> trueRelations = new HashSet<>();

    for(KBPSlotFill triple : triples) {
      if( triple.key.slotValue.equals(pair.slotValue) )
        trueRelations.add(triple.key.relationName);
    }

    return trueRelations;
  }

  /**
   * Ask the knowledge pair if it knows about any valid slots between given
   * entity pair.
   * @param entity The entity we are finding slot fills for
   */
  @Override
  public List<KBPSlotFill> getKnownSlotFillsForEntity(KBPEntity entity) {
    if (knowledgeBase.data.isEmpty() && !Props.HACKS_DONTREADKB) {
      warn("getting relations without having loaded KB! Loading training KB by default");
      this.trainingTriples();
    }
    List<KBPSlotFill> fills = new ArrayList<>(knowledgeBase.get(entity).getOrElse(new HashSet<>()));
    Collections.sort(fills, (a, b) -> a.key.relationName.compareTo(b.key.relationName));
    return fills;
  }

  @SuppressWarnings("unchecked")
  @Override
  public IterableIterator<Pair<Annotation, Double>> queryKeywords(Collection<String> words, Maybe<Integer> maxDocs) {
    if (Props.INDEX_MODE != Props.QueryMode.NOOP ) {
      return officialIndex.queryKeywords(words, maxDocs);
    } else {
      return new IterableIterator<Pair<Annotation,Double>>(Collections.EMPTY_LIST.iterator());
    }
  }

}
