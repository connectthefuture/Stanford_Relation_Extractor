package edu.stanford.nlp.kbp.slotfilling.ir;

import edu.stanford.nlp.ie.machinereading.structure.EntityMention;
import edu.stanford.nlp.ie.machinereading.structure.MachineReadingAnnotations;
import edu.stanford.nlp.ie.machinereading.structure.RelationMention;
import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.kbp.slotfilling.classify.HeuristicRelationExtractor;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.ProtobufAnnotationSerializer;
import edu.stanford.nlp.time.TimeAnnotations;
import edu.stanford.nlp.util.ArrayCoreMap;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.logging.PrettyLoggable;
import edu.stanford.nlp.util.logging.Redwood;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import static edu.stanford.nlp.util.logging.Redwood.Util.debug;
import static edu.stanford.nlp.util.logging.Redwood.Util.RED;

/**
 * This class tracks the provenance of a particular entity of interest -- usually a Slot fill.
 *
 * @author Gabor Angeli
 */
public class KBPRelationProvenance implements Serializable {
  private static final long serialVersionUID = 1L;

  /** The document id we have found */
  public final String docId;
  /** The index name we searched in */
  public final String indexName;
  /** The index of the sentence in the document relevant to the relation */
  public final Maybe<Integer> sentenceIndex;
  /** The span of the entity in the sentence */
  public final Maybe<Span> entityMentionInSentence;
  /** The span of the slot fill in the sentence */
  public final Maybe<Span> slotValueMentionInSentence;
  /** Optionally, the sentence in which this slot fill occurs */
  public Maybe<CoreMap> containingSentenceLossy;

  /** Confidence score assigned by the classifier */
  public final Maybe<Double> score;

  public Maybe<Class> classifierClass = Maybe.Nothing();

  public KBPRelationProvenance(String docId, String indexName) {
    this.docId = docId != null ? docId.trim() : null;
    this.indexName = indexName != null ? indexName.trim() : null;
    this.sentenceIndex = Maybe.Nothing();
    this.entityMentionInSentence = Maybe.Nothing();
    this.slotValueMentionInSentence = Maybe.Nothing();
    this.containingSentenceLossy = Maybe.Nothing();
    if (isOfficial()) {
      debug("Provenance", "creating official provenance without sentence level information");
    }
    this.score = Maybe.Nothing();
    assert this.docId != null;
    assert this.indexName != null;
  }

  public KBPRelationProvenance(String docId, String indexName, int sentenceIndex, Span entityMention, Span slotValue,
                               CoreMap containingSentence, Maybe<Double> score, Class classifierClass) {
      this(docId, indexName, sentenceIndex, entityMention, slotValue, containingSentence, score);
      setClassifierClass(classifierClass);
  }

  public KBPRelationProvenance(String docId, String indexName, int sentenceIndex, Span entityMention, Span slotValue,
                               CoreMap containingSentence, Maybe<Double> score) {
    this.docId = docId != null ? docId.trim() : null;
    this.indexName = indexName != null ? indexName.trim() : null;
    this.sentenceIndex = Maybe.Just(sentenceIndex);
    this.entityMentionInSentence = Maybe.Just(entityMention);
    this.slotValueMentionInSentence = Maybe.Just(slotValue);
    assert slotValue != null;
    // Save some useful information from the containing sentence.
    // Be careful: a bunch of these get serialized as datums.
    CoreMap lossySentence;
    if (Props.KBP_VALIDATE && Props.VALIDATE_RULES_DO) {
      lossySentence = containingSentence;  // in validation mode, this is our only hook into the original sentence
    } else {

      lossySentence = new ArrayCoreMap(1);
  	if(containingSentence!=null){
      List<CoreLabel> tokens = new ArrayList<>(containingSentence.get(CoreAnnotations.TokensAnnotation.class).size());
      for (CoreLabel token : containingSentence.get(CoreAnnotations.TokensAnnotation.class)) {
        CoreLabel lossyToken = new CoreLabel(6);
        lossyToken.set(CoreAnnotations.TextAnnotation.class, token.get(CoreAnnotations.TextAnnotation.class));
        lossyToken.set(CoreAnnotations.AntecedentAnnotation.class, token.get(CoreAnnotations.AntecedentAnnotation.class));
        lossyToken.set(CoreAnnotations.PartOfSpeechAnnotation.class, token.get(CoreAnnotations.PartOfSpeechAnnotation.class));
        lossyToken.set(CoreAnnotations.NamedEntityTagAnnotation.class, token.get(CoreAnnotations.NamedEntityTagAnnotation.class));
        lossyToken.set(CoreAnnotations.NumericValueAnnotation.class, token.get(CoreAnnotations.NumericValueAnnotation.class));
        lossyToken.set(TimeAnnotations.TimexAnnotation.class, token.get(TimeAnnotations.TimexAnnotation.class));
        tokens.add(lossyToken);      
      }
      lossySentence.set(CoreAnnotations.TokensAnnotation.class, tokens);
  	}
    }
    this.containingSentenceLossy = Maybe.Just(lossySentence);
    this.score = score;
  }

  public KBPRelationProvenance(String docId, String indexName, int sentenceIndex, Span entityMention, Span slotValue,
                               CoreMap containingSentence) {
    this(docId, indexName, sentenceIndex, entityMention, slotValue, containingSentence, Maybe.<Double>Nothing());
  }

  public KBPRelationProvenance(String docId, String indexName, int sentenceIndex, Span entityMention, Span slotValue) {
    this(docId, indexName, Maybe.Just(sentenceIndex), Maybe.Just(entityMention), Maybe.Just(slotValue), Maybe.<CoreMap>Nothing(), Maybe.<Double>Nothing());
  }

  private KBPRelationProvenance(String docId, String indexName, Maybe<Integer> sentenceIndex,
                               Maybe<Span> entityMention, Maybe<Span> slotValue,
                               Maybe<CoreMap> containingSentence, Maybe<Double> score) {
    this.docId = docId != null ? docId.trim() : null;
    this.indexName = indexName != null ? indexName.trim() : null;
    this.sentenceIndex = sentenceIndex;
    this.entityMentionInSentence = entityMention;
    this.slotValueMentionInSentence = slotValue;
    this.containingSentenceLossy = containingSentence;
    this.score = score;
  }

  /**
   * Convert this object to its corresponding Protocol Buffer.
   */
  public KBPProtos.KBPRelationProvenance toProto() {
    KBPProtos.KBPRelationProvenance.Builder builder = KBPProtos.KBPRelationProvenance.newBuilder();
    builder
        .setDocid(this.docId)
        .setIndexName(this.indexName);
    for (int x : this.sentenceIndex) { builder.setSentenceIndex(x); }
    for (Span x : this.entityMentionInSentence) { builder.setEntityMention(KBPProtos.Span.newBuilder().setStart(x.start()).setEnd(x.end()).build()); }
    for (Span x : this.slotValueMentionInSentence) { builder.setSlotValueMention(KBPProtos.Span.newBuilder().setStart(x.start()).setEnd(x.end()).build()); }
    for (CoreMap sentence : this.containingSentenceLossy) {
      ProtobufAnnotationSerializer serializer = new ProtobufAnnotationSerializer(false);
      builder.setContainingSentence(serializer.toProto(sentence));
    }
    for (Double x : this.score) { builder.setScore(x); }
    return builder.build();
  }

  /**
   * Read a provenance from a protocol buffer.
   * @param proto The protocol buffer to read.
   * @return A {@link edu.stanford.nlp.kbp.slotfilling.ir.KBPRelationProvenance} object built from the proto.
   */
  public static KBPRelationProvenance fromProto(KBPProtos.KBPRelationProvenance proto) {
    String docid = proto.getDocid();
    String indexName = proto.getIndexName();
    Maybe<Integer> sentenceIndex = proto.hasSentenceIndex() ? Maybe.Just(proto.getSentenceIndex()) : Maybe.<Integer>Nothing();
    Maybe<Span> entityMentionInSentence
        = proto.hasEntityMention() ? Maybe.Just(new Span(proto.getEntityMention().getStart(), proto.getEntityMention().getEnd())) : Maybe.<Span>Nothing();
    Maybe<Span> slotMentionInSentence
        = proto.hasSlotValueMention() ? Maybe.Just(new Span(proto.getSlotValueMention().getStart(), proto.getSlotValueMention().getEnd())) : Maybe.<Span>Nothing();
    Maybe<CoreMap> containingSentence = Maybe.Nothing();
    if (proto.hasContainingSentence()) {
      ProtobufAnnotationSerializer serializer = new ProtobufAnnotationSerializer();
      containingSentence = Maybe.Just(serializer.fromProto(proto.getContainingSentence()));
    }
    Maybe<Double> score = proto.hasScore() ? Maybe.Just(proto.getScore()) : Maybe.<Double>Nothing();
    return new KBPRelationProvenance(docid, indexName, sentenceIndex, entityMentionInSentence, slotMentionInSentence, containingSentence, score);
  }

  public String gloss(final KBPIR querier) {
    if (sentenceIndex.isDefined() && entityMentionInSentence.isDefined() && slotValueMentionInSentence.isDefined()) {
      try {
        // Get offsets
        if (!(querier instanceof StandardIR)) {
          throw new IllegalArgumentException("Not sure how to fetch documents by docid through anything but a KBPIR");
        }
        Maybe<Annotation> doc;
        if (Props.INDEX_MODE == Props.QueryMode.NOOP) {
          doc = Maybe.Nothing();
        } else {
          doc = ((StandardIR) querier).officialIndex.fetchDocument(docId);
        }
        if (doc.isDefined()) {
          String text = doc.get().get(CoreAnnotations.TextAnnotation.class);
          CoreMap sentence = doc.get().get(CoreAnnotations.SentencesAnnotation.class).get(sentenceIndex.get());
          return CoreMapUtils.sentenceToProvenanceString(text, sentence, entityMentionInSentence.get(), slotValueMentionInSentence.get()).replaceAll("\n", " ").replaceAll("\\s+", " ");
        }
      } catch (IOException e) {
        throw new RuntimeException(e);
      } catch (RuntimeException ignored) {
      }
    }
    return ("  <no sentence provenance>");
  }

  public PrettyLoggable loggable(final KBPIR querier) {
    return (channels, description) -> {
      if (sentenceIndex.isDefined() && entityMentionInSentence.isDefined() && slotValueMentionInSentence.isDefined()) {
        try {
          // Get offsets
          if (querier instanceof StandardIR) {
            Maybe<Annotation> doc;
            if (Props.INDEX_MODE != Props.QueryMode.NOOP) {
              doc  = ((StandardIR) querier).officialIndex.fetchDocument(docId);
            }  else {
              doc = Maybe.Nothing();
            }
            if (doc.isDefined()) {
              String text = doc.get().get(CoreAnnotations.TextAnnotation.class);
              CoreMap sentence = doc.get().get(CoreAnnotations.SentencesAnnotation.class).get(sentenceIndex.get());
              channels.log("         " + CoreMapUtils.sentenceToProvenanceString(text, sentence, entityMentionInSentence.get(), slotValueMentionInSentence.get()).replaceAll("\n", " ").replaceAll("\\s+", " "));
            } else {
              channels.err(RED, "         could not fetch document!");
            }
          }
        } catch (IOException e) {
          throw new RuntimeException(e);
        } catch (RuntimeException e) {
          channels.log(e);
        }
      } else {
        channels.log("  <no sentence provenance>");
      }
    };
  }

  /**
   * Convert this provenance into a slot context to be used in entity linking.
   * @param entity The entity tho link in this provenance.
   * @param ir A potential IR component to retrieve sentences from.
   * @return An entity context, if available, for this provenance.
   */
  public Maybe<EntityContext> toSlotContext(KBPEntity entity, Maybe<KBPIR> ir) {
    if (!ir.isDefined()) { return Maybe.Nothing(); }
    if (!sentenceIndex.isDefined()) { return Maybe.Nothing(); }
    if (!slotValueMentionInSentence.isDefined()) { return Maybe.Nothing(); }
    Annotation doc = ir.get().fetchDocument(this.docId);
    List<CoreMap> sentences = doc.get(CoreAnnotations.SentencesAnnotation.class);
    return Maybe.Just(new EntityContext(entity, sentences.get(sentenceIndex.get()),
        new Span(slotValueMentionInSentence.get().start(), slotValueMentionInSentence.get().end())));
  }

  /**
   * Return the distance between the entity and the slot fill, in tokens.
   */
  public Maybe<Integer> distanceBetweenEntities() {
    for (Span entitySpan : entityMentionInSentence) {
      //noinspection LoopStatementThatDoesntLoop
      for (Span slotSpan : slotValueMentionInSentence) {
        return Maybe.Just(Span.distance(entitySpan, slotSpan));
      }
    }
    return Maybe.Nothing();
  }

  /**
   * Compute a relation provenance from a sentence and KBTriple.
   * If possible, this will also populate the in-sentence provenance.
   * @param sentence The sentence we are using for provenance
   * @param key The KBTriple representing the entities we would like to obtain provenance on
   * @return A provenance object with as much information filled in as possible
   */
  public static Maybe<KBPRelationProvenance> compute(CoreMap sentence, KBTriple key) {
    String docid = sentence.get(CoreAnnotations.DocIDAnnotation.class);
    assert docid != null;
    String indexName = sentence.get(KBPAnnotations.SourceIndexAnnotation.class);
    assert indexName != null;

    KBPRelationProvenance rtn = null;
    List<RelationMention> relationMentions = sentence.get(MachineReadingAnnotations.RelationMentionsAnnotation.class);
    if (relationMentions == null) { relationMentions = new LinkedList<>(); }
    for (RelationMention rel : relationMentions) {
      assert (rel instanceof NormalizedRelationMention);
      NormalizedRelationMention normRel = (NormalizedRelationMention) rel;
      KBPEntity entity = normRel.getNormalizedEntity();
      String slotValue = normRel.getNormalizedSlotValue();
      List<EntityMention> args = normRel.getEntityMentionArgs();
      EntityMention leftArg = args.get(0);
      EntityMention rightArg = args.get(1);
      if ((entity == null || key.getEntity().equals(entity)) && slotValue.trim().equalsIgnoreCase(key.slotValue.trim())) {
        Integer sentenceIndex = leftArg.getSentence().get(CoreAnnotations.SentenceIndexAnnotation.class);
        Span entitySpan = leftArg.getExtent();
        Span slotValueSpan = rightArg.getExtent();
        if (sentenceIndex != null) {
          rtn = new KBPRelationProvenance(docid, indexName, sentenceIndex, entitySpan, slotValueSpan, sentence);
        }
      }
    }

    return Maybe.fromNull(rtn);
  }
  
  public static Maybe<KBPRelationProvenance> computeFromSpans(CoreMap sentence, Span entitySpan, Span slotValueSpan) {
    String docid = sentence.get(CoreAnnotations.DocIDAnnotation.class);
    assert docid != null;
    String indexName = sentence.get(KBPAnnotations.SourceIndexAnnotation.class);
    assert indexName != null;
    Integer sentenceIndex = sentence.get(CoreAnnotations.SentenceIndexAnnotation.class);
    KBPRelationProvenance rtn = null;
    if (sentenceIndex != null) {
      rtn = new KBPRelationProvenance(docid, indexName, sentenceIndex, entitySpan, slotValueSpan, sentence);
    }
      
    return Maybe.fromNull(rtn);
  }

  public boolean isOfficial() {
    if (!isOfficialIndex(this.indexName)) { return false; }
    // TODO(gabor) this is a bit of an awful hack, and should maybe be removed
    if (this.entityMentionInSentence.isDefined() && this.slotValueMentionInSentence.isDefined()) {
      if (Span.distance(this.entityMentionInSentence.get(), this.slotValueMentionInSentence.get()) > Props.MAX_DISTANCE_BETWEEN_ENTITY_AND_SLOT) {
        return false;
      }
    }
    return true;
  }

  public static boolean isOfficialIndex(String indexName) {
    //noinspection SimplifiableIfStatement
    if (Props.INDEX_OFFICIAL == null) { return false; } // for JUnit tests; this should never be null
    return indexName != null && indexName.toLowerCase().endsWith(Props.INDEX_OFFICIAL.getName().toLowerCase());
  }

  public KBPRelationProvenance rewrite(double score) {
    return new KBPRelationProvenance(docId, indexName, sentenceIndex.get(), entityMentionInSentence.get(), slotValueMentionInSentence.get(), containingSentenceLossy.get(), Maybe.Just(score));
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (!(o instanceof KBPRelationProvenance)) return false;
    KBPRelationProvenance that = (KBPRelationProvenance) o;
    return docId.equals(that.docId) && entityMentionInSentence.equals(that.entityMentionInSentence) && indexName.equals(that.indexName) && sentenceIndex.equals(that.sentenceIndex) && slotValueMentionInSentence.equals(that.slotValueMentionInSentence);

  }

  @Override
  public int hashCode() {
    int result = docId.hashCode();
    result = 31 * result + indexName.hashCode();
    result = 31 * result + sentenceIndex.hashCode();
    result = 31 * result + entityMentionInSentence.hashCode();
    result = 31 * result + slotValueMentionInSentence.hashCode();
    return result;
  }

  @Override
  public String toString() {
    return "KBPRelationProvenance{" +
        "docId='" + docId + '\'' +
        ", indexName='" + indexName + '\'' +
        ", sentenceIndex=" + sentenceIndex +
        ", entityMentionInSentence=" + entityMentionInSentence +
        ", slotValueMentionInSentence=" + slotValueMentionInSentence +
        '}';
  }

  public void setClassifierClass(Class classifierClass) {
    if(classifierClass!= null){
    this.classifierClass = Maybe.Just(classifierClass);
    }else
      this.classifierClass = Maybe.Nothing();
  }

  public Maybe<Class> getClassifierClass() {
    return this.classifierClass;
  }
}
