package edu.stanford.nlp.kbp.slotfilling.ir;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IterableIterator;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.logging.Redwood;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Stream;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * An interface for querying documents and sentences, with many utility methods, and a few
 * key methods which should be implemented.
 *
 * @author Gabor Angeli
 */
@SuppressWarnings("UnusedDeclaration")
public abstract class KBPIR {

  protected static final Redwood.RedwoodChannels logger = Redwood.channels("IR");

  //
  // To Override
  //

  /**
   * The all-encompasing query method -- most things are really syntactic sugar on top
   * of this method.
   *
   * Note that from here on, the convention of putting relation after slotValue disappears, as every method will take
   * both a relation and a slot fill.
   *
   * @param tableName The table to write to (either document or sentence)
   * @param expectedOutput The class to output to. This is either a CoreMap (for sentences) or Annotation (for documents)
   * @param entity The entity to query
   * @param slotValue   The slot fill value we are querying for (e.g., Scotland)
   * @param docidsToForce A set of docids to always query
   * @param maxDocuments The maximum number of documents to search over. This is not necessarily the same as the number
   * @param officialIndexOnly If set to true, the query will only run on the official index.
   * @return A sorted list of the top sentences
   */
  protected abstract <E extends CoreMap> List<E> queryCoreMaps(final String tableName, final Class<E> expectedOutput,
                                                               final KBPEntity entity, final Maybe<KBPEntity> slotValue,
                                                               final Maybe<String> relation, final Set<String> docidsToForce,
                                                               final int maxDocuments, final boolean officialIndexOnly);

  /**
   * Fetch a document from all the indices.
   */
  public abstract Annotation fetchDocument(String docId, boolean officialIndexOnly);

  /**
   * Get the total number of documents containing a set of search terms, in any document, as defined by its
   * corresponding querier.
   *
   * @param terms The query phrases to search for.
   * @return The number of results in the index.
   */
  public abstract int queryNumHits(Collection<String> terms);

  /**
   * Get all documents known the the IR index.
   * These should be processed to conform to the contract of {@link PostIRAnnotator}, and be tagged
   * with their source index, etc.
   * @return A lazy iterator over documents to be read.
   */
  public abstract Stream<Annotation> slurpDocuments(int numDocuments);

  public Stream<Annotation> slurpDocuments(){
    return slurpDocuments(Integer.MAX_VALUE);
  }


  /** The top level function to query for sentences */
  protected List<CoreMap> querySentences(KBPEntity entity, Maybe<KBPEntity> slotValue,
                                         Maybe<String> relation,
                                         Set<String> docidsToForce,
                                         int maxDocuments,
                                         boolean officialIndexOnly) {
    return queryCoreMaps(Props.DB_TABLE_SENTENCE_CACHE, CoreMap.class, entity, slotValue, relation, docidsToForce, maxDocuments, officialIndexOnly);
  }

  /** The top level function to query for entire documents */
  protected List<Annotation> queryDocuments(KBPEntity entity,
                                            Maybe<KBPEntity> slotValue,
                                            Maybe<String> relation,
                                            Set<String> docidsToForce,
                                            int maxDocuments,
                                            boolean officialIndexOnly) {
    return queryCoreMaps(Props.DB_TABLE_DOCUMENT_CACHE, Annotation.class, entity, slotValue, relation, docidsToForce, maxDocuments, officialIndexOnly);
  }

  /**
   * Similar to querySentences, but only returns KBP DocIDs, and doesn't hit the CoreMap.
   *
   * @see KBPIR#querySentences(KBPEntity, Maybe, Maybe, Set, int, boolean) querySentences
   */
  protected abstract List<String> queryDocIDs(String entityName,
                                              Maybe<NERTag> entityType,
                                              Maybe<String> relation,
                                              Maybe<String> slotValue,
                                              Maybe<NERTag> slotValueType,
                                              int maxDocuments,
                                              boolean officialIndexOnly);

  public abstract Set<String> getKnownRelationsForPair(KBPair pair);
  public abstract List<KBPSlotFill> getKnownSlotFillsForEntity(KBPEntity entity);


  //
  // Shared Data
  //

  /**
   * A cached version of the knowledge base.
   * This is to allow reading it from a simple serialized file, rather than having to parse
   * the XML each time.
   */
  protected static KnowledgeBase knowledgeBase = new KnowledgeBase();

  /**
   * Retrieve the knowledge base. Note that unlike {@link edu.stanford.nlp.kbp.slotfilling.ir.KBPIR#trainingTriples()},
   * this method will return the entire known knowledge base, not limited by {@link Props#TRAIN_TUPLES_COUNT}.
   */
  public synchronized KnowledgeBase getKnowledgeBase() {
    forceTrack("Reading KB");
    if (knowledgeBase == null || knowledgeBase.isEmpty()) {
      // Knowledge base is not cached -- start caching it
      for (KBTriple tuple: trainingDataFromKBPTSV(Integer.MAX_VALUE, Props.TRAIN_TUPLES_FILES)) {
        knowledgeBase.put(KBPNew.from(tuple).KBPSlotFill());
      }
      for (KBTriple tuple: trainingDataFromTSV(Integer.MAX_VALUE, Props.TRAIN_TUPLES_AUX)) {
        knowledgeBase.put(KBPNew.from(tuple).KBPSlotFill());
      }
    }
    endTrack("Reading KB");
    return knowledgeBase;
  }

  //
  // Helper if not already there
  //

  /**
   * A function to read a knowledge base from the simplest possible triple store, storing triples of the form:
   * <code>
   *   entityName \t entityType \t relation \t slotValue \t slotType
   * </code>
   * @param limit The maximum entries to read
   * @param files The files to read from
   * @return A knowledge base composed of the KBTriples read from the file.
   */
  public List<KBTriple> trainingDataFromTSV(int limit, String... files) {
    List<KBTriple> kb = new ArrayList<>();
    int numRead = 0;
    for (String file : files) {
      for (String line : IOUtils.linesFromFile(file)) {
        String[] fields = line.split("\t");
        kb.add(KBPNew.entName(fields[0]).entType(fields[1]).slotValue(fields[3]).slotType(fields[4]).rel(fields[2]).KBTriple());
        numRead += 1;
        if (numRead > limit) { return kb; }
      }
      logger.log("read " + kb.size() + " triples from " + file);
    }
    return kb;
  }


  /**
   * Get the list of training tuples, from a list of files
   * Once the limit is reached, the tuples are returned
   * @param limit The number of tuples to read, or -1 to read all available tupels
   * @param files The TSV files to read the tuples from.
   * @return A list of KBTriple objects, corresponding to the (entity, relation, slotValue) triples found in the TSV file.
   */
  public List<KBTriple> trainingDataFromKBPTSV(int limit, String... files) {
    List<KBTriple> tuples = new ArrayList<>();
    for (String file:files) {
      readTuplesFromKBPTSV(tuples, file, limit);
    }
    return tuples;
  }

  /** The tab character */
  private static final Pattern TAB_PATTERN = Pattern.compile("\\t");
  /**
   * Get the list of training tuples, from one TSV file.
   * Format of the file is tab delimited with fields: entityId, entityName, relationName, and slotValue
   * @return A list of KBTriple objects, corresponding to the (entity, relation, slotValue) triples found in the TSV file.
   */
  private List<KBTriple> readTuplesFromKBPTSV(List<KBTriple> tuples, String filename, int limit) {
    try {
      // Slurp file
      BufferedReader bufferedReader = IOUtils.getBufferedFileReader(filename);
      String line;
      List<String[]> lines = new ArrayList<>();
      while ((line = bufferedReader.readLine()) != null) {
        lines.add(TAB_PATTERN.split(line));
      }

      // Compute entity NER votes
      Map<String, Counter<NERTag>> entityNERCounts = new HashMap<>();
      for (String[] fields : lines) {
        String entityName = fields[1];
        if (!entityNERCounts.containsKey(entityName)) { entityNERCounts.put(entityName, new ClassicCounter<NERTag>()); }
        NERTag entType = NERTag.fromRelation(fields[2]).orCrash("Unknown relation " + fields[2]);
        entityNERCounts.get(entityName).incrementCount(entType);
      }
      Map<String, NERTag> entityNERTags = new HashMap<>();
      for (Map.Entry<String, Counter<NERTag>> entry : entityNERCounts.entrySet()) {
        entityNERTags.put(entry.getKey(), Counters.argmax(entry.getValue()));
      }

      int count = 0;
      boolean tuplesCountReached = false;
      for (String[] fields : lines) {
        count++;
        if (fields.length == 4) {
          // 0 is entityId, 1 is entityName, 2 is relationName, 3 is slotValue
          String entityId = fields[0];
          String entityName = fields[1];
          NERTag entType = entityNERTags.get(entityName);
          String relationName = fields[2];
          String slotValue = fields[3];
          KBTriple triple = KBPNew.entName(entityName).entType(entType).entId(entityId).slotValue(slotValue).rel(relationName).KBTriple();

          // Check if this tuple type checks
          for (RelationType relation : triple.tryKbpRelation()) {
            if (relation.entityType != triple.entityType || // if doesn't match entity type (e.g., a per: relation for an ORG)
                (triple.slotType.isDefined() &&  // ... or doesn't match slot type (with fudge allowed for non-regexner tags, in case it hasn't been run)
                    triple.slotType.get().isRegexNERType && !relation.validNamedEntityLabels.contains(triple.slotType.get())) ) {
              logger.debug("invalid KB entry: " + triple);
            } else {
              tuples.add(triple);
            }
          }

          if (limit > 0 && tuples.size() >= limit) {
            tuplesCountReached = true;
            break;
          }
        } else {
          throw new RuntimeException("Error reading tuples from TSV: Invalid line at " + filename + ":" + count);
        }
      }
      logger.log("read " + tuples.size() + " triples from " + filename + ((tuplesCountReached) ? " reached tupled count" : ""));
      bufferedReader.close();
      return tuples;
    } catch (IOException ex) {
      throw new RuntimeException("Error reading tuples from " + filename, ex);
    }
  }

  /**
   * Get the knowledge base as a list of triples, taking only the number of triples
   * specified in {@link Props#TRAIN_TUPLES_COUNT}.
   */
  public List<KBTriple> trainingTriples() {
    List<KBTriple> triples = getKnowledgeBase().triples();
    return triples.subList(0, Math.min(triples.size(), Props.TRAIN_TUPLES_COUNT));
  }

  /**
   * Get the tuples in the knowledge base, deduplicated and in a reliable order, to be used for training
   * instances. Note that these don't contain relation annotations, and therefore are primarily for use in
   * IR.
   * @return A list of {@link KBPair}s corresponding to the training triples (e.g., Obama, Hawaii)
   */
  public List<KBPair> trainingTuples() {
    Set<KBPair> seenPairs = new HashSet<>();
    List<KBPair> pairs = new ArrayList<>();
    for (KBTriple triple : trainingTriples()) {
      KBPair pair = KBPNew.from(triple).KBPair();
      if (seenPairs.add(pair)) {
        pairs.add(pair);
      }
    }
    return pairs;
  }

  /**
   * Returns known slot values for a given entity and relation.
   * For example, Barack Obama and born_in should return Set(Hawaii).
   * @param entity The entity to query
   * @param rel The relation to fill with the slot fills
   * @return A set of known slot fills for the given entity and relation.
   */
  public Set<String> getKnownSlotValuesForEntityAndRelation(KBPEntity entity, RelationType rel) {
    List<KBPSlotFill> triples = getKnownSlotFillsForEntity(entity);
    Set<String> slotValues = new HashSet<>();
    for (KBPSlotFill triple : triples) {
      if (triple.key.relationName.equals(rel.canonicalName)) {
        slotValues.add(triple.key.slotValue);
      }
    }
    return slotValues;
  }

  public List<CoreMap> querySentences(KBPEntity entity, Maybe<KBPEntity> slotValue, int n, boolean officialIndexOnly) {
    // Get IR retreivals
    List<CoreMap> sentences = querySentences(entity, slotValue, Maybe.<String>Nothing(), new HashSet<String>(), n, officialIndexOnly);
    // TODO(gabor) this is kind of a hack
    // In case the cache is stale, force the source index to be the correct index
    if (officialIndexOnly) {
      for (CoreMap sentence : sentences) {
        if (sentence.containsKey(CoreAnnotations.SentenceIndexAnnotation.class)) {
          sentence.set(KBPAnnotations.SourceIndexAnnotation.class, Props.INDEX_OFFICIAL.getPath());
        }
      }
    }
    // Return
    return sentences;
  }

  public List<CoreMap> querySentences(KBPEntity entity, Maybe<KBPEntity> slotValue, int n) {
    return querySentences(entity, slotValue, n, false);
  }

  public List<CoreMap> querySentences(KBPEntity entity, int n) {
    return querySentences(entity, Maybe.<KBPEntity>Nothing(), n, false);
  }

  public List<CoreMap> querySentences(KBPEntity entity, KBPEntity slotValue, int n) {
    return querySentences(entity, Maybe.Just(slotValue), n, false);
  }

  public List<CoreMap> querySentences( KBPEntity entity, Set<String> docidsToForce, int n  ) {
    return querySentences(entity, Maybe.<KBPEntity>Nothing(), Maybe.<String>Nothing(), docidsToForce, n, false);
  }

  public List<CoreMap> querySentences( String entityName, String slotValue, int n  ) {
    return querySentences(KBPNew.entName(entityName).entType(NERTag.PERSON).KBPEntity(),
        Maybe.Just(KBPNew.entName(slotValue).entType(NERTag.MISC).KBPEntity()), n, false);
  }

  @SuppressWarnings("unchecked")
  public List<CoreMap> querySentences( String entityName, String slotValue, Maybe<String> relationName, int n  ) {
    return querySentences(KBPNew.entName(entityName).entType(NERTag.PERSON).KBPEntity(),
        Maybe.Just(KBPNew.entName(slotValue).entType(NERTag.MISC).KBPEntity()),
        relationName, Collections.EMPTY_SET, n, false);
  }

  public List<String> queryDocIDs( String entityName, String slotValue, String reln, int n  ) {
    return queryDocIDs(entityName, Maybe.<NERTag>Nothing(),
        Maybe.Just(reln), Maybe.Just(slotValue),
        Maybe.<NERTag>Nothing(), n, false);
  }

  public List<String> queryDocIDs( String entityName, String slotValue, int n  ) {
    return queryDocIDs(entityName, Maybe.<NERTag>Nothing(),
        Maybe.<String>Nothing(), Maybe.Just(slotValue),
        Maybe.<NERTag>Nothing(), n, false);
  }

  public List<String> queryDocIDs(String entityName, RelationType relation, int maxDocuments) {
    return queryDocIDs(entityName, Maybe.<NERTag>Nothing(),
        Maybe.Just(relation.canonicalName), Maybe.<String>Nothing(),
        Maybe.<NERTag>Nothing(), maxDocuments, false);
  }

  public List<String> queryDocIDs(String entityName, NERTag entityType, int maxDocuments) {
    return queryDocIDs(entityName, Maybe.Just(entityType),
            Maybe.<String>Nothing(), Maybe.<String>Nothing(),
            Maybe.<NERTag>Nothing(), maxDocuments, false);
  }

  public List<String> queryDocIDs(String entityName, NERTag entityType, Set<String> docidsToForce, int maxDocuments) {
    // Add the forced docids
    List<String> rtn = new ArrayList<>(docidsToForce);
    // Add the queried docids
    List<String> docs = queryDocIDs(entityName, Maybe.Just(entityType),
        Maybe.<String>Nothing(), Maybe.<String>Nothing(),
        Maybe.<NERTag>Nothing(), maxDocuments, false);
    for (String doc : docs) {
      if (!docidsToForce.contains(doc)) {
        rtn.add(doc);
      }
    }
    return rtn;
  }

  public List<Annotation> queryDocuments(KBPEntity entity, int maxDocuments) {
    return queryDocuments(entity, Maybe.<KBPEntity>Nothing(), Maybe.<String>Nothing(), new HashSet<String>(), maxDocuments, false);
  }

  public List<Annotation> queryDocuments(KBPEntity entity, Set<String> docidsToForce, int maxDocuments) {
    return queryDocuments(entity, Maybe.<KBPEntity>Nothing(), Maybe.<String>Nothing(), docidsToForce, maxDocuments, false);
  }

  public Annotation fetchDocument(String docId) {
    return fetchDocument( docId, false );
  }

  protected static KBTriple toKBTriple( Pair<String, String> pair, String relation ) {
    NERTag entityType = NERTag.fromRelation( relation ).orCrash();
    return KBPNew.entName(pair.first).entType(entityType).entId(Maybe.<String>Nothing()).slotValue(pair.second).rel(relation).KBTriple();
  }

  public abstract IterableIterator<Pair<Annotation, Double>> queryKeywords(Collection<String> words, Maybe<Integer> maxDocs);
}
