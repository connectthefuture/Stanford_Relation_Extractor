package edu.stanford.nlp.kbp.slotfilling;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

import edu.stanford.nlp.ie.machinereading.structure.EntityMention;
import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPRelationProvenance;
import edu.stanford.nlp.kbp.slotfilling.ir.PostIRAnnotator;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.AnnotationPipeline;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.pipeline.TokensRegexNERAnnotator;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.*;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * <p>A collection of top-level tasks one may wish to do with a slotfilling system.
 * For example, classify the relation for a raw sentence or coremap, or fill the slots
 * for an entity (in String form). </p>
 *
 * <p>The key unifying principle for all methods in this class is that they should be independent
 * of the specific KBP task, and rather be used either for interfacing with external services
 * (e.g., the grant deliverables), or wi unit tests (which don't care about KBP per se). </p>
 *
 * @author Gabor Angeli
 */
@SuppressWarnings("UnusedDeclaration")
public class SlotfillingTasks {
  public final SlotfillingSystem system;

  public SlotfillingTasks(SlotfillingSystem system) {
    this.system = system;
  }

  private Maybe<StanfordCoreNLP> pipeline = Maybe.Nothing();
  public StanfordCoreNLP pipeline() {
    //noinspection LoopStatementThatDoesntLoop
    for (StanfordCoreNLP p : pipeline) { return p; }
    Properties coreNLPProperties = new Properties();
    if (!system.props.isEmpty()) {
      for(Object key : system.props.keySet()) {
        if (system.props.getProperty((String) key) != null && key != null) {
          coreNLPProperties.put(key, system.props.get(key));
        }
      }
    }
    coreNLPProperties.put("annotators", Props.ANNOTATORS);
    pipeline = Maybe.Just(new StanfordCoreNLP(coreNLPProperties));
    return pipeline();
  }

  public final AnnotationPipeline onlineNER = new AnnotationPipeline() {{
    addAnnotator(new TokensRegexNERAnnotator(Props.PROCESS_REGEXNER_DIR.getPath() + File.separator + Props.PROCESS_REGEXNER_CASELESS, true, "^(NN|JJ).*"));
    addAnnotator(new TokensRegexNERAnnotator(Props.PROCESS_REGEXNER_DIR.getPath() + File.separator + Props.PROCESS_REGEXNER_WITHCASE, false, null));
  }};

  /**
   * Fill slots extracted from a single sentence.
   * Note that no processing of the slots is done at this stage.
   * @param entity The entity for whom we are extracting slots
   * @param sentenceGloss The sentence we are extracting slots from
   * @return A set of Slot Fill objects for this entity in this sentence.
   */
  public Set<KBPSlotFill> getSlotsInSentence(String entity, String sentenceGloss){
    CoreMap sentence = stringToCoreMap(entity, sentenceGloss);
    Set<KBPSlotFill> fills = new HashSet<KBPSlotFill>();
    for (SentenceGroup datum : sentenceToClassifierInput(entity, sentence)) {
      Counter<Pair<String,Maybe<KBPRelationProvenance>>> relations
          = this.system.getTrainedClassifier().get().classifyRelations(datum, Maybe.<CoreMap[]>Nothing());
      for (Map.Entry<Pair<String, Maybe<KBPRelationProvenance>>, Double> relationBlob : relations.entrySet()) {
        fills.add(KBPNew.from(datum.key).rel(relationBlob.getKey().first).provenance(relationBlob.getKey().second).KBPSlotFill());
      }
    }
    return fills;
  }

  /**
   * Classify a sentence (as a CoreMap)into its corresponding relation.
   * @param entity The KBPOfficialEntity form of the entity we are pivoting on
   * @param sentence The CoreNLP annotated CoreMap we are classifying.
   * @return The distribution over possible relations
   */
  public Counter<String> classifyRelation(KBPOfficialEntity entity, CoreMap sentence, String slotValue) {
    Maybe<SentenceGroup> toClassify = sentenceToClassifierInput(entity, sentence, slotValue);
    if (!toClassify.isDefined()) {
      warn("No datum found for " + entity + " with slot fill " + slotValue);
      return new ClassicCounter<String>();
    } else {
      return system.getTrainedClassifier().get().classifyRelationsNoProvenance(toClassify.get(), Maybe.Just(new CoreMap[]{sentence}));
    }
  }

  /**
   * Classify a sentence (as a CoreMap)into its corresponding relation.
   * @param entity The String gloss of the entity we are pivoting on
   * @param sentence The CoreNLP annotated CoreMap we are classifying.
   * @return The relation this sentence expresses
   */
  public Counter<String> classifyRelation(String entity, CoreMap sentence) {
    return classifyRelation(stringToEntity(entity), sentence, null);
  }

  /**
   * Classify a sentence into its corresponding relation.
   * @param entity The pivot entity in the sentence
   * @param sentence The sentence we are classifying
   * @return The relation this sentence expresses
   */
  public Counter<String> classifyRelation(String entity, String sentence) {
    return classifyRelation(entity, stringToCoreMap(entity, sentence));
  }

  /**
   * Classify a sentence (as a CoreMap)into its corresponding relation.
   * @param entity The String gloss of the entity we are pivoting on
   * @param sentence The CoreNLP annotated CoreMap we are classifying.
   * @param slotValue The value of the slot we would like to classify.
   * @return The relation this sentence expresses
   */
  public Counter<String> classifyRelation(String entity, CoreMap sentence, String slotValue) {
    return classifyRelation(stringToEntity(entity), sentence, slotValue);
  }

  /**
   * Classify a sentence (as a CoreMap)into its corresponding relation.
   * @param entity The String gloss of the entity we are pivoting on
   * @param sentence The CoreNLP annotated CoreMap we are classifying.
   * @param slotValue The value of the slot we would like to classify.
   * @return The relation this sentence expresses
   */
  public Counter<String> classifyRelation(String entity, String sentence, String slotValue) {
    return classifyRelation(entity, stringToCoreMap(entity, sentence), slotValue);
  }

  /**
   * Convert a sentence (and entity to pivot on) into a SentenceGroup to be passed to the relation classifer.
   * @param entity The entity we are pivoting on
   * @param sentence The sentence we would like to convert into a sentence group
   * @return A sentence group, ready to be fed into the classifier.
   */
  public Collection<SentenceGroup> sentenceToClassifierInput(KBPOfficialEntity entity, CoreMap sentence) {
    // Annotate
    List<CoreMap> sentences = new LinkedList<CoreMap>();
    sentences.add(sentence);
    sentences = system.getProcess().annotateSentenceFeatures(entity, sentences);
    // Featurize
    Annotation annotation = new Annotation("");
    annotation.set(CoreAnnotations.SentencesAnnotation.class, sentences);
    Map<KBPair,SentenceGroup> datums = system.getProcess().featurize(annotation);
    // Return
    return datums.values();
  }

  /**
   * @see SlotfillingTasks#sentenceToClassifierInput(KBPOfficialEntity, CoreMap)
   */
  public Collection<SentenceGroup> sentenceToClassifierInput(KBPOfficialEntity entity, String sentence) {
    return sentenceToClassifierInput(entity, stringToCoreMap(entity.name, sentence));
  }

  /**
   * @see SlotfillingTasks#sentenceToClassifierInput(KBPOfficialEntity, CoreMap)
   */
  public Collection<SentenceGroup> sentenceToClassifierInput(String entity, String sentence) {
    return sentenceToClassifierInput(stringToEntity(entity), stringToCoreMap(entity, sentence));
  }

  /**
   * @see SlotfillingTasks#sentenceToClassifierInput(KBPOfficialEntity, CoreMap)
   */
  public Collection<SentenceGroup> sentenceToClassifierInput(String entity, CoreMap sentence) {
    return sentenceToClassifierInput(stringToEntity(entity), sentence);
  }


  /**
   * Create a single sentence group from an entity and sentence, corresponding to the specific slot value.
   *
   * @see SlotfillingTasks#sentenceToClassifierInput(KBPOfficialEntity, CoreMap)
   */
  public Maybe<SentenceGroup> sentenceToClassifierInput(KBPOfficialEntity entity, CoreMap sentence, String slotValue) {
    Collection<SentenceGroup> datums = sentenceToClassifierInput(entity, sentence);
    SentenceGroup toClassify = null;
    for (SentenceGroup datum : datums) {
      if (slotValue == null || datum.key.slotValue.equalsIgnoreCase(slotValue) || datum.key.slotValue.equalsIgnoreCase(slotValue.replaceAll("_", " "))) {
        toClassify = datum;
        break;
      }
    }
    if (toClassify == null) return Maybe.Nothing(); else return Maybe.Just(toClassify);
  }

  /**
   * @see SlotfillingTasks#sentenceToClassifierInput(KBPOfficialEntity, CoreMap, String)
   */
  public Maybe<SentenceGroup> sentenceToClassifierInput(KBPOfficialEntity entity, String sentence, String slotValue) {
    return sentenceToClassifierInput(entity, stringToCoreMap(entity.name, sentence, slotValue), slotValue);
  }

  /**
   * @see SlotfillingTasks#sentenceToClassifierInput(KBPOfficialEntity, CoreMap, String)
   */
  public Maybe<SentenceGroup> sentenceToClassifierInput(String entity, CoreMap sentence, String slotValue) {
    return sentenceToClassifierInput(stringToEntity(entity), sentence, slotValue);
  }

  /**
   * @see SlotfillingTasks#sentenceToClassifierInput(KBPOfficialEntity, CoreMap, String)
  */
  public Maybe<SentenceGroup> sentenceToClassifierInput(String entity, String sentence, String slotValue) {
    return sentenceToClassifierInput(stringToEntity(entity), sentence, slotValue);
  }

  /**
   * Featurize a String sentence into a CoreMap representing the sentence
   * @param sentence The sentence gloss to featurize.
   * @param entityName An optional entity name to annotate
   * @param slotValue An optional slot value to annotate
   * @return A CoreMap corresponding to the first sentence of the [dummy] document we annotated.
   */
  public CoreMap stringToRawCoreMap(String sentence, Maybe<String> entityName, Maybe<String> slotValue) {
    // Create CoreMap
    Annotation doc = new Annotation(sentence);
    pipeline().annotate(doc);
    if (entityName.isDefined()) {
      if (slotValue.isDefined()) {
        new PostIRAnnotator(KBPNew.entName(entityName.get()).entType(NERTag.PERSON).KBPOfficialEntity(),
            KBPNew.entName(slotValue.get()).entType(NERTag.PERSON).KBPEntity(), true).annotate(doc);
      } else {
        new PostIRAnnotator(KBPNew.entName(entityName.get()).entType(NERTag.PERSON).KBPOfficialEntity(), true).annotate(doc);
      }
    }
    // Online RegexNER
    onlineNER.annotate(doc);
    // Get Sentence
    if (doc.get(CoreAnnotations.SentencesAnnotation.class).size() > 1) {
      warn("Multiple sentences extracted (taking only first): " + sentence);
    }
    return doc.get(CoreAnnotations.SentencesAnnotation.class).get(0);
  }

  /**
   * Featurize a String sentence into a CoreMap representing the sentence
   * @param sentence The sentence gloss to featurize.
   * @return A CoreMap corresponding to the first sentence of the [dummy] document we annotated.
   */
  public CoreMap stringToRawCoreMap(String sentence) {
    return stringToRawCoreMap(sentence, Maybe.<String>Nothing(), Maybe.<String>Nothing());
  }

  /**
   * Featurize a String sentence into a CoreMap representing the sentence,
   * with annotations for the entity and explicit slot fill added (note that the slot fill being added is a bit hacky)
   * @param entity The entity to annotate
   * @param sentence The sentence gloss to featurize.
   * @param slotValues The slot values to annotate in addition to the entity
   * @return A CoreMap corresponding to the first sentence of the [dummy] document we annotated.
   */
  public CoreMap stringToCoreMap(String entity, String sentence, String... slotValues) {
    CoreMap raw = stringToRawCoreMap(sentence, Maybe.Just(entity), slotValues.length > 0 ? Maybe.Just(slotValues[0]) : Maybe.<String>Nothing());
    // Annotate
    List<CoreMap> sentences = new LinkedList<CoreMap>();
    sentences.add(raw);
    List<CoreMap> processedSentences = system.getProcess().annotateSentenceFeatures(stringToEntity(entity), sentences);
    // Process
    CoreMap processed = processedSentences.size() > 0 ? processedSentences.get(0) : sentences.get(0);
    // Add slot value mentions  // TODO(gabor) this could be less hacky...
    int i = 0;
    for (String slotValue : slotValues) {
      int start = -1;
      for (CoreLabel token : processed.get(CoreAnnotations.TokensAnnotation.class)) {
        if (slotValue.startsWith(token.originalText())) {  start = i; }
        if (slotValue.endsWith(token.originalText())) {
          EntityMention toAdd = new EntityMention("noID", processed, new Span(start, i + 1), new Span(start, i + 1), token.ner(), null, null);
          if (!processed.containsKey(KBPAnnotations.SlotMentionsAnnotation.class)) { processed.set(KBPAnnotations.SlotMentionsAnnotation.class, new ArrayList<EntityMention>()); }
          processed.get(KBPAnnotations.SlotMentionsAnnotation.class).add(toAdd);
        }
        i += 1;
      }
    }
    // Return
    return processed;
  }

  /**
   * Featurize a String sentence into a CoreMap representing the sentence,
   * with annotations for the entity and slot fill added
   * @param entity The entity to annotate
   * @param sentence The sentence gloss to featurize.
   * @return A CoreMap corresponding to the first sentence of the [dummy] document we annotated.
   */
  public CoreMap stringToCoreMap(String entity, String sentence) {
    return stringToCoreMap(entity, sentence, new String[0]);
  }

  /**
   * Convert a String to a KBPOfficialEntity, trying to automatically fill in the entity type.
   * @param entityString The KBP Entity as a String
   * @return A KBPOfficialEntity object representing this String
   */
  public KBPOfficialEntity stringToEntity(String entityString) {
    // Get NER annotation
    Annotation singlePhrase = new Annotation(entityString);
    pipeline().annotate(singlePhrase);
    List<CoreLabel> tokens = singlePhrase.get(CoreAnnotations.SentencesAnnotation.class).get(0).get(CoreAnnotations.TokensAnnotation.class);
    NERTag entityType = null;
    for (CoreLabel token : tokens) {
      if (entityType == null && token.get(CoreAnnotations.NamedEntityTagAnnotation.class).equalsIgnoreCase("PERSON")) {
        entityType = NERTag.PERSON;
      }
      if (entityType == null && token.get(CoreAnnotations.NamedEntityTagAnnotation.class).equalsIgnoreCase("ORGANIZATION")) {
        entityType = NERTag.ORGANIZATION;
      }
    }
    // Ensure the type is not null
    if (entityType == null) {
      warn(RED, "could not find NER tag for " + entityString + " -- defaulting to ORG");
      entityType = NERTag.ORGANIZATION;
    }
    // Create Entity
    return stringToEntity(entityString, entityType);
  }

  /**
   * Convert a String to a KBPOfficialEntity, with a specific entity type.
   * This method should be syntactic sugar for new KBPOfficialEntity(string, entitytype)
   * @param entityString The KBP Entity as a String
   * @param entityType The type of the entity (PER, ORG, NIL)
   * @return A KBPOfficialEntity object representing this String
   */
  public KBPOfficialEntity stringToEntity(String entityString, NERTag entityType) {
    return KBPNew.entName(entityString).entType(entityType).KBPOfficialEntity();
  }

  /**
   * Create a Sentence Group from a tokenized sentence, and associated spans
   * @param sentence The tokenized sentence
   * @param entitySpan The span for the entity mention
   * @param slotFillSpan The span for the slot fill mention
   * @return A sentence group, corresponding to this featurized sentence
   */
  public Maybe<SentenceGroup> sentenceToClassifierInput(String[] sentence, Span entitySpan, Span slotFillSpan) {
    StringBuilder entityName = new StringBuilder();
    for (int i = entitySpan.start(); i < entitySpan.end(); ++i) {
      entityName.append(sentence[i]).append(" ");
    }
    StringBuilder slotFillName = new StringBuilder();
    for (int i = slotFillSpan.start(); i < slotFillSpan.end(); ++i) {
      slotFillName.append(sentence[i]).append(" ");
    }

    KBPOfficialEntity entity = KBPNew.entName(entityName.toString()).entType(NERTag.PERSON).KBPOfficialEntity();
    return sentenceToClassifierInput(entity, StringUtils.join(sentence, " "), slotFillName.toString().trim());
  }

  /**
   * Start a console, asking for an entity name, and then printing all the slots known for that entity
   */
  public void console() throws IOException {
    // Don't save anything
    Props.CACHE_SENTENCES_DO = false;
    Props.CACHE_DATUMS_DO = false;
    Props.CACHE_GRAPH_DO = false;
    Props.CACHE_SENTENCEGLOSS_DO = false;
    Props.CACHE_PROVENANCE_DO = false;
    Props.TEST_PROVENANCE_DO = true;  // this makes things much faster

    // Start the console
    log ("--- CONSOLE STARTED ---");
    log ("");
    int queryIndex = 0;
    Index<String> entityIDs = new HashIndex<String>();
    while (true) {
      // Read input
      log("");
      log("Enter an query of the form '<entity_name>:<entity_type>':");
      String queryString = new BufferedReader(new InputStreamReader(System.in)).readLine();

      // Error check
      if (queryString.equalsIgnoreCase("exit") || queryString.equalsIgnoreCase("quit")) {
        break;
      }

      // Parse entity
      if (!queryString.contains(":")) {
        err("Query format: <entity_name>:<entity_type>");
        continue;
      }
      String entityName = queryString.substring(0, queryString.indexOf(":"));
      Maybe<NERTag> entityType = NERTag.fromString(queryString.substring(queryString.indexOf(":") + 1).trim());
      if (!entityType.isDefined()) {
        err("Could not parse entity type: " + queryString.substring(queryString.indexOf(":")));
        continue;
      }
      KBPOfficialEntity queryEntity = KBPNew.entName(entityName.trim()).entType(entityType.orCrash())
          .queryId("CUSTOM" + queryIndex).entId("" + entityIDs.addToIndex(queryString)).KBPOfficialEntity();
      queryIndex += 1;

      // Run Query
      List<KBPSlotFill> slotFills = system.getEvaluator().getSlotFiller().fillSlots(queryEntity);
      forceTrack("Slot Fills");
      for (KBPSlotFill slotFill : slotFills) { log(slotFill); }
      log("(" + slotFills.size() + " total)");
      endTrack("Slot Fills");

    }
  }
}
