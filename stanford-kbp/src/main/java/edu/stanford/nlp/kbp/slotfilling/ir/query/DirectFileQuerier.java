package edu.stanford.nlp.kbp.slotfilling.ir.query;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.kbp.slotfilling.ir.PostIRAnnotator;
import edu.stanford.nlp.kbp.slotfilling.ir.index.KryoAnnotationSerializer;
import edu.stanford.nlp.kbp.slotfilling.ir.webqueries.RelationMentionSnippets;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.AnnotationSerializer;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IterableIterator;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.logging.Redwood;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.*;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

/**
 * A querier which simply looks for sentences directly in a serialized file,
 * rather than querying through Lucene.
 *
 * The file to look for is defined in {@link DirectFileQuerier#query2filename(String, edu.stanford.nlp.kbp.common.Maybe, edu.stanford.nlp.kbp.common.Maybe, edu.stanford.nlp.kbp.slotfilling.ir.webqueries.RelationMentionSnippets.QueryType)}
 *
 * @author Gabor Angeli
 */
public class DirectFileQuerier implements Querier {

  protected static final Redwood.RedwoodChannels logger = Redwood.channels("FileIR");

  public final File rootDirectory;

  public DirectFileQuerier(File rootDirectory) {
    if (!rootDirectory.exists() || !rootDirectory.canRead() || !rootDirectory.isDirectory()) {
      throw new IllegalArgumentException("Cannot create querier at " + rootDirectory + " (cannot read directory)");
    }
    this.rootDirectory = rootDirectory;
  }

  @Override
  public IterableIterator<Pair<CoreMap, Double>> querySentences(final KBPEntity entity,
        final Maybe<KBPEntity> slotValue,
        final Maybe<String> relationName,
        final Set<String> docidsToForce,
        final Maybe<Integer> maxDocuments) {
    final PostIRAnnotator postIRAnnotator;
    if (slotValue.isDefined()) {
      postIRAnnotator = new PostIRAnnotator(entity instanceof KBPOfficialEntity ? (KBPOfficialEntity) entity : KBPNew.from(entity).KBPOfficialEntity(), slotValue.get(), true);
    } else {
      postIRAnnotator = new PostIRAnnotator(entity instanceof KBPOfficialEntity ? (KBPOfficialEntity) entity : KBPNew.from(entity).KBPOfficialEntity(), true);

    }
    return CollectionUtils.flatMapIgnoreNull(Arrays.asList(RelationMentionSnippets.QueryType.values()).iterator(), in -> {
      Maybe<String> slotValueName = slotValue.isDefined() ? Maybe.Just(slotValue.get().name) : Maybe.<String>Nothing();
      Pair<String, String> fileInfo = query2filename(entity.name, relationName, slotValueName, in);
      File toRead = new File(Props.INDEX_WEBSNIPPETS_DIR + File.separator + fileInfo.first + File.separator + fileInfo.second);
      if (toRead.exists()) {
        try {

          // vv Code Starts Here vv
          Pair<Annotation, InputStream> pair = serializer.read(new FileInputStream(toRead));
          pair.second.close();
          List<CoreMap> sentences = pair.first.get(CoreAnnotations.SentencesAnnotation.class);
          List<Pair<CoreMap, Double>> sentencesWithScore = new ArrayList<>();
          for (CoreMap sent : sentences) {
            // Annotate sentence
            Annotation ann = new Annotation(sent.get(CoreAnnotations.TextAnnotation.class));
            ann.set(CoreAnnotations.SentencesAnnotation.class, Arrays.asList(sent));
            ann.set(CoreAnnotations.TokensAnnotation.class, sent.get(CoreAnnotations.TokensAnnotation.class));
            postIRAnnotator.annotate(ann);
            // Check for entity + slot fill mentions
            Set<String> antecedents = sent.get(KBPAnnotations.AllAntecedentsAnnotation.class);
            if (antecedents != null && antecedents.contains(entity.name) &&  // must contain entity name
                (!slotValue.isDefined() || antecedents.contains(slotValue.get().name) || sent.get(CoreAnnotations.TextAnnotation.class).contains(slotValue.get().name))) {  // must contain slot value (if defined)
              sentencesWithScore.add(Pair.makePair(sent, 1.0));
            }
          }
          logger.log("found " + sentencesWithScore.size() + " (of " + sentences.size() + " possible) websnippets");
          return sentencesWithScore.iterator();
          // ^^ Code Ends Here ^^

        } catch (IOException e) {
          logger.log(e);
          return new ArrayList<Pair<CoreMap, Double>>(0).iterator();
        } catch (ClassNotFoundException e) {
          logger.log(e);
          return new ArrayList<Pair<CoreMap, Double>>(0).iterator();
        }
      } else {
        logger.log("no websnippets for query");
        return new ArrayList<Pair<CoreMap, Double>>(0).iterator();
      }
    });
  }

  /** {@inheritDoc} */
  @Override
  public Stream<Annotation> slurp(int maxDocuments) {
    if(maxDocuments < Integer.MAX_VALUE)
      logger.err("Max Documents not implemented for DirectFileQuerier");

    final PostIRAnnotator dummyPostIR = new PostIRAnnotator(KBPNew.entName("__NO ENTITY__").entType(NERTag.PERSON).KBPOfficialEntity());

    return StreamSupport.stream(
        Spliterators.spliteratorUnknownSize(
            IOUtils.iterFilesRecursive(this.rootDirectory).iterator(), Spliterator.ORDERED | Spliterator.CONCURRENT), true)
        .map( toRead -> {
          try {
            Pair<Annotation, InputStream> pair = serializer.read(new FileInputStream(toRead));
            pair.second.close();
            pair.first.set(KBPAnnotations.SourceIndexAnnotation.class, "file://" + rootDirectory.getPath());
            for (CoreMap sentence : pair.first.get(CoreAnnotations.SentencesAnnotation.class)) {
              sentence.set(KBPAnnotations.SourceIndexAnnotation.class, "file://" + rootDirectory.getPath());
            }
            dummyPostIR.annotate(pair.first);
            return pair.first;
          } catch (IOException e) {
            logger.err(e);
            return null;
          } catch (ClassNotFoundException e) {
            logger.err(e);
            return null;
          }
        })
        .filter( ann -> ann != null );
  }

  @Override
  public void close() { }


  private static String toAlphaNumeric(String input) {
    return input
        .replaceAll("\\s+", "_")             // no spaces
        .replaceAll(File.separator, "SLASH") // no slashes
        .replaceAll("#", "POUND")            // no pound signs (this would be nice to leave as the field separator)
        .replaceAll("[^\\x00-\\x7F]", "")    // don't include non-ascii
        .replaceAll("[^a-zA-Z0-9_]", "-")    // no other special characters
        .toLowerCase();
  }
  /**
   * A deterministic mapping from query information to a file to search for.
   * Note that this function will not attempt to backoff. For example, if a file exists with
   * only the entityName defined, but the query specifies a slot fill, then the filename returned will be
   * that of the (nonexistant) file specified by the (entity, slot fill) pair.
   *
   * @param entityName The entity to query for
   * @param relation The relation, as per RelationType.name (not to be confused with RelationType.name())
   * @param slotValue The slot value
   * @param queryType The query type. A separate file is created for each query type.
   * @return A subdirectory and filename corresponding the the serialized websnippet sentences for this query
   */
  public static Pair<String, String> query2filename(String entityName, Maybe<String> relation, Maybe<String> slotValue, RelationMentionSnippets.QueryType queryType) {
    assert entityName != null;
    assert  relation != null && relation.getOrElse("") != null;
    assert  slotValue != null && slotValue.getOrElse("") != null;
    assert  queryType != null;
    StringBuilder b = new StringBuilder();
    b.append(queryType.name());
    // entity
    b.append("#");
    b.append(toAlphaNumeric(entityName));
    // relation
    b.append("#");
    for (String r : relation) { b.append(toAlphaNumeric(r));}
    // slot fill
    b.append("#");
    for (String v : slotValue) { b.append(toAlphaNumeric(v)); }
    // extension
    b.append(".ser.gz");
    return Pair.makePair(b.substring(0, 6), b.toString());
  }

  /** The serializer implementation being used to save and load files */
  public static final AnnotationSerializer serializer = new KryoAnnotationSerializer(true, true, true);

  @Override
  /** {@inheritDoc} */
  public String toString() { return this.rootDirectory.getPath(); }
  @Override
  /** {@inheritDoc} */
  public int hashCode() { return this.rootDirectory.hashCode(); }
  @Override
  /** {@inheritDoc} */
  public boolean equals(Object o) {
    return o instanceof DirectFileQuerier && ((DirectFileQuerier) o).rootDirectory.equals(this.rootDirectory);
  }
}
