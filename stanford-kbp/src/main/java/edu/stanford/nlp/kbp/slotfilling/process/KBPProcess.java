package edu.stanford.nlp.kbp.slotfilling.process;

import static edu.stanford.nlp.util.logging.Redwood.Util.err;

import java.sql.Connection;
import java.sql.SQLException;
import java.util.*;

import edu.stanford.nlp.ie.machinereading.structure.*;
import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.kbp.common.KBPAnnotations.SourceIndexAnnotation;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPIR;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPRelationProvenance;
import edu.stanford.nlp.kbp.slotfilling.ir.index.KryoAnnotationSerializer;
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.ling.CoreAnnotations.DocIDAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.AnnotationPipeline;
import edu.stanford.nlp.pipeline.AnnotationSerializer;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.logging.Redwood;

/**
 * <p>The entry point for processing and featurizing a sentence or group of sentences.
 * In particular, before being fed to the classifier either at training or evaluation time,
 * a sentence should be passed through the {@link KBPProcess#annotateSentenceFeatures(KBPEntity, List)}
 * function, as well as the {@link KBPProcess#featurizeSentence(CoreMap, Maybe)}
 * (or variations of this function, e.g., the buik {@link KBPProcess#featurize(Annotation)}).</p>
 *
 * <p>At a high level, this function is a mapping from {@link CoreMap}s representing sentences to
 * {@link SentenceGroup}s representing datums (more precisely, a collection of datums for a single entity pair).
 * The input should have already passed through {@link edu.stanford.nlp.kbp.slotfilling.ir.PostIRAnnotator}; the
 * output is ready to be passed into the classifier.</p>
 *
 * <p>This is also the class where sentence gloss caching is managed. That is, every datum carries with itself a
 * hashed "sentence gloss key" which alleviates the need to carry around the raw sentence, but can be used to retrieve
 * that sentence if it is needed -- primarily, for Active Learning. See {@link KBPProcess#saveSentenceGloss(String, CoreMap, Maybe, Maybe)}
 * and {@link KBPProcess#recoverSentenceGloss(String)}.</p>
 *
 * @author Kevin Reschke
 * @author Jean Wu (sentence gloss infrastructure)
 * @author Gabor Angeli (managing hooks into Mention Featurizers; some hooks into sentence gloss caching)
 */
public class KBPProcess extends Featurizer implements DocumentAnnotator {

  protected static final Redwood.RedwoodChannels logger = Redwood.channels("Process");

  private final FeatureFactory rff;

  public static final AnnotationSerializer sentenceGlossSerializer = new KryoAnnotationSerializer();
  private final Properties props;  // needed to create a StanfordCoreNLP down the line
  private final Lazy<KBPIR> querier;

  public enum AnnotateMode { 
                              NORMAL,   // do normal relation annotation for main entity
                              ALL_PAIRS  // do relation annotation for all entity pairs
                           }
  
  public KBPProcess(Properties props, Lazy<KBPIR> querier) {
    this.props = props;
    this.querier = querier;
    // Setup feature factory
    rff = new FeatureFactory(Props.TRAIN_FEATURES);
    rff.setDoNotLexicalizeFirstArgument(true);
  }

  @SuppressWarnings("unchecked")
  public Maybe<Datum<String,String>> featurize( RelationMention rel ) {
    try {
      if (!Props.PROCESS_NEWFEATURIZER) {
        // Case: Old featurizer
        return Maybe.Just(rff.createDatum(rel));
      } else {
        // Case: New featurizer
        Span subj = ((EntityMention) rel.getArg(0)).getHead();
        Span obj = ((EntityMention) rel.getArg(1)).getHead();
        List<CoreLabel> tokens = rel.getSentence().get(CoreAnnotations.TokensAnnotation.class);
        SemanticGraph dependencies = rel.getSentence().get(SemanticGraphCoreAnnotations.BasicDependenciesAnnotation.class);
        Featurizable factory = new Featurizable(subj, obj, tokens, dependencies, Collections.EMPTY_LIST);
        Counter<String> features = new ClassicCounter<>();
        for (Feature feat : Feature.values()) {
          feat.provider.apply(factory, features);
        }
        // Return
        return Maybe.<Datum<String, String>>Just(new BasicDatum<>(Counters.toSortedList(features), rel.getType()));
      }
    } catch (RuntimeException e) {
      err(e);
      return Maybe.Nothing();
    }
  }

  /**
   * Featurize |sentence| with respect to |entity|, with optional |filter|.
   * 
   * That is, build a datum (singleton sentence group) for each relation that
   * is headed by |entity|, subject to filtering.
   *  
   * @param sentence
   *          CoreMap with relation mention annotations. 
   * @param filter
   *          Optional relation filter for within-sentence filtering
   * @return List of singleton sentence groups.  Each singleton represents a
   *          datum for one of the relations found in this sentence.
   */
  @Override
  public List<SentenceGroup> featurizeSentence(CoreMap sentence, Maybe<RelationFilter> filter) {
    List<RelationMention> relationMentionsForEntity = sentence.get(MachineReadingAnnotations.RelationMentionsAnnotation.class);
    List<RelationMention> relationMentionsForAllPairs = sentence.get(MachineReadingAnnotations.AllRelationMentionsAnnotation.class);

    List<SentenceGroup> datumsForEntity = featurizeRelations(relationMentionsForEntity,sentence);
    
    if(filter.isDefined()) {
      List<SentenceGroup> datumsForAllPairs = featurizeRelations(relationMentionsForAllPairs,sentence);
      datumsForEntity = filter.get().apply(datumsForEntity, datumsForAllPairs, sentence);
    }
    
    return datumsForEntity;
  }

  /**
   * Construct datums for these relation mentions.
   * Output is list of singleton sentence groups.
   */
  private List<SentenceGroup> featurizeRelations(List<RelationMention> relationMentions, CoreMap sentence) {
    List<SentenceGroup> datums = new ArrayList<>();
    if (relationMentions == null) { return datums; }

    for (RelationMention rel : relationMentions) {
      assert rel instanceof NormalizedRelationMention;
      NormalizedRelationMention normRel = (NormalizedRelationMention) rel;
      assert normRel.getEntityMentionArgs().get(0).getSyntacticHeadTokenPosition() >= 0;
      assert normRel.getEntityMentionArgs().get(1).getSyntacticHeadTokenPosition() >= 0;
      for (Datum<String, String> d : featurize(rel)) {


        // Pull out the arguments to construct the entity pair this
        // datum will express.
        List<EntityMention> args = rel.getEntityMentionArgs();
        EntityMention leftArg = args.get(0);
        EntityMention rightArg = args.get(1);
        KBPEntity entity = normRel.getNormalizedEntity();
        String slotValue = normRel.getNormalizedSlotValue();

        // Create key
        KBPair key = KBPNew
            .entName(entity != null ? entity.name : (leftArg.getNormalizedName() != null ? leftArg.getNormalizedName() : leftArg.getFullValue()))
            .entType(entity != null ? entity.type : Utils.getNERTag(leftArg).orCrash())
            .entId(entity != null && entity instanceof KBPOfficialEntity ? ((KBPOfficialEntity) entity).id : Maybe.<String>Nothing())
            .slotValue(slotValue)
            .slotType(Utils.getNERTag(rightArg).orCrash()).KBPair();
        logger.debug("featurized datum key: " + key);

        // Also track the provenance information
        String indexName = leftArg.getSentence().get(SourceIndexAnnotation.class);
        String docId = leftArg.getSentence().get(DocIDAnnotation.class);
        Integer sentenceIndex = leftArg.getSentence().get(CoreAnnotations.SentenceIndexAnnotation.class);
        Span entitySpan = leftArg.getExtent();
        Span slotFillSpan = rightArg.getExtent();
        KBPRelationProvenance provenance =
            sentenceIndex == null ? new KBPRelationProvenance(docId, indexName)
                : new KBPRelationProvenance(docId, indexName, sentenceIndex, entitySpan, slotFillSpan, sentence);

        // Handle Sentence Gloss Caching
        String hexKey = CoreMapUtils.getSentenceGlossKey(sentence.get(CoreAnnotations.TokensAnnotation.class), leftArg.getExtent(), rightArg.getExtent());
        saveSentenceGloss(hexKey, sentence, Maybe.Just(entitySpan), Maybe.Just(slotFillSpan));

        // Construct singleton sentence group; group by slotValue entity
        SentenceGroup sg = new SentenceGroup(key, d, provenance, hexKey);
        datums.add(sg);
      }
    }
    return datums;
  }

  @Override
  public List<CoreMap> annotateSentenceFeatures (KBPEntity entity, List<CoreMap> sentences) {
    return annotateSentenceFeatures(entity,sentences,AnnotateMode.NORMAL);
  }
  
  public List<CoreMap> annotateSentenceFeatures( KBPEntity entity,
                                                 List<CoreMap> sentences, AnnotateMode annotateMode) {
    // Check if PostIR was run
    for (CoreMap sentence : sentences) {
      if (!sentence.containsKey(KBPAnnotations.AllAntecedentsAnnotation.class) && !Props.JUNIT) {
        throw new IllegalStateException("Must pass sentence through PostIRAnnotator before calling AnnotateSentenceFeatures");
      }
    }

    // Create the mention annotation pipeline
    AnnotationPipeline pipeline = new AnnotationPipeline();
    pipeline.addAnnotator(new EntityMentionAnnotator(entity));
    pipeline.addAnnotator(new SlotMentionAnnotator());
    pipeline.addAnnotator(new RelationMentionAnnotator(entity, querier.get().getKnownSlotFillsForEntity(entity), annotateMode));
    pipeline.addAnnotator(new PreFeaturizerAnnotator(props));
    // Annotate
    Annotation ann = new Annotation(sentences);
    pipeline.annotate(ann);
    // Sanity checks
    if (Utils.assertionsEnabled()) {
      for (CoreMap sentence : ann.get(SentencesAnnotation.class)) {
        for (RelationMention rm : sentence.get(MachineReadingAnnotations.RelationMentionsAnnotation.class)) {
          assert rm.getArg(0) instanceof EntityMention;
          assert rm.getArg(1) instanceof EntityMention;
          assert ((EntityMention) rm.getArg(0)).getSyntacticHeadTokenPosition() >= 0;
          assert ((EntityMention) rm.getArg(1)).getSyntacticHeadTokenPosition() >= 0;
        }
      }
    }
    // Return valid sentences
    return ann.get(SentencesAnnotation.class);
  }

  public void saveSentenceGloss(final String hexKey, CoreMap sentence, Maybe<Span> entitySpanMaybe, Maybe<Span> slotFillSpanMaybe) {
    if (Props.CACHE_SENTENCEGLOSS_DO) {
      assert (!hexKey.isEmpty());
      // Create annotation
      final Annotation ann = new Annotation("");
      ann.set(SentencesAnnotation.class, new ArrayList<>(Arrays.asList(sentence)));
      // Set extra annotations
      for (Span entitySpan : entitySpanMaybe) {
        sentence.set(KBPAnnotations.EntitySpanAnnotation.class, entitySpan);  // include entity and slot value spans
      }
      for (Span slotFillSpan : slotFillSpanMaybe) {
        sentence.set(KBPAnnotations.SlotValueSpanAnnotation.class, slotFillSpan);
      }
      // Do caching
      PostgresUtils.withKeyAnnotationTable(Props.DB_TABLE_SENTENCEGLOSS_CACHE, new PostgresUtils.KeyAnnotationCallback(sentenceGlossSerializer) {
        @Override
        public void apply(Connection psql) throws SQLException {
          try {
            putSingle(psql, Props.DB_TABLE_SENTENCEGLOSS_CACHE, hexKey, ann);
          } catch (SQLException e) {
            logger.err(e);
          }
        }
      });
      // Clear extra annotations
      sentence.remove(KBPAnnotations.EntitySpanAnnotation.class);  // don't keep these around indefinitely -- they're datum not sentence specific
      sentence.remove(KBPAnnotations.SlotValueSpanAnnotation.class);
    }
  }

  /** Recovers the original sentence, given a short hash pointing to the sentence */
  public Maybe<CoreMap> recoverSentenceGloss(final String hexKey) {
    final Pointer<CoreMap> sentence = new Pointer<>();
    PostgresUtils.withKeyAnnotationTable(Props.DB_TABLE_SENTENCEGLOSS_CACHE, new PostgresUtils.KeyAnnotationCallback(sentenceGlossSerializer) {
      @Override
      public void apply(Connection psql) throws SQLException {
        for (Annotation ann : getSingle(psql, Props.DB_TABLE_SENTENCEGLOSS_CACHE, hexKey)) {
          for (CoreMap sent : Maybe.fromNull(ann.get(SentencesAnnotation.class).get(0))) {
            sentence.set(sent);
          }
        }
      }
    });
    return sentence.dereference();
  }

}
