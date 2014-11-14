package edu.stanford.nlp.kbp.common;

import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.ProtobufAnnotationSerializer;
import edu.stanford.nlp.util.CoreMap;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * The context an entity appears in, or as much of it as is known.
 *
 * @author Gabor Angeli
 */
public class EntityContext {
  /** The entity corresponding to this context. This is required to be defined */
  public final KBPEntity entity;

  /**
   * The sentence in which this entity occurs.
   * If this is defined, {@link EntityContext#entityTokenSpan} is also defined.
   */
  public final Maybe<CoreMap> sentence;
  /**
   * The span that this entity fills in the sentence.
   * If this is defined, {@link EntityContext#sentence} is also defined.
   */
  public final Maybe<Span> entityTokenSpan;

  /**
   * The document this entity occurs in.
   * If this is defined, {@link EntityContext#sentenceIndex},
   * {@link EntityContext#sentence} and {@link EntityContext#entityTokenSpan} are also defined.
   */
  public final Maybe<Annotation> document;
  /**
   * The index of the sentence the entity mention occurs in, relative to the document.
   * If this is defined, {@link EntityContext#document},
   * {@link EntityContext#sentence} and {@link EntityContext#entityTokenSpan} are also defined.
   */
  public final Maybe<Integer> sentenceIndex;

  /**
   * Known properties of the entity, as a collection of slot fills.
   * This may be defined independently of other fields in this class.
   */
  public final Maybe<Collection<KBPSlotFill>> properties;


  /**
   * Create a new entity context, without any context.
   * @param entity The entity whose context we're representing.
   */
  public EntityContext(KBPEntity entity) {
    this.entity = entity;
    this.sentence = Maybe.Nothing();
    this.entityTokenSpan = Maybe.Nothing();
    this.document = Maybe.Nothing();
    this.sentenceIndex = Maybe.Nothing();
    this.properties = Maybe.Nothing();
  }

  /**
   * Create a new entity context: the entity, context in which it's found.
   * @param entity The entity whose context we're representing.
   * @param contextSentence The document in which this entity occurs.
   * @param contextSpan The span within the document in which this entity occurs.
   */
  public EntityContext(KBPEntity entity, CoreMap contextSentence, Span contextSpan) {
    this.entity = entity;
    this.sentence = Maybe.Just(contextSentence);
    this.entityTokenSpan = Maybe.Just(contextSpan);
    this.document = Maybe.Nothing();
    this.sentenceIndex = Maybe.Nothing();
    this.properties = Maybe.Nothing();
  }

  /**
   * Create a new entity context: the entity, and known properties of the entity.
   * @param entity The entity whose context we're representing.
   * @param properties Known properties of the entity.
   */
  public EntityContext(KBPEntity entity,Collection<KBPSlotFill> properties) {
    this.entity = entity;
    this.sentence = Maybe.Nothing();
    this.entityTokenSpan = Maybe.Nothing();
    this.document = Maybe.Nothing();
    this.sentenceIndex = Maybe.Nothing();
    this.properties = Maybe.Just(properties);
  }

  /**
   * Create a new entity context: the entity, context in which it's found, and known properties of the entity.
   * @param entity The entity whose context we're representing.
   * @param contextSentence The sentence in which this entity occurs.
   * @param contextSpan The span within the document in which this entity occurs.
   * @param properties Known properties of the entity.
   */
  public EntityContext(KBPEntity entity, CoreMap contextSentence, Span contextSpan, Collection<KBPSlotFill> properties) {
    this.entity = entity;
    this.sentence = Maybe.Just(contextSentence);
    this.entityTokenSpan = Maybe.Just(contextSpan);
    this.document = Maybe.Nothing();
    this.sentenceIndex = Maybe.Nothing();
    this.properties = Maybe.Just(properties);
  }

  /**
   * Create a new entity context: the entity, context in which it's found.
   * @param entity The entity whose context we're representing.
   * @param contextDocument The document in which this entity occurs.
   * @param sentenceIndex The index of the sentence the entity mention occurs in.
   * @param contextSpan The span within the document in which this entity occurs.
   */
  public EntityContext(KBPEntity entity, Annotation contextDocument, int sentenceIndex, Span contextSpan) {
    this.entity = entity;
    this.sentence = Maybe.Just(contextDocument.get(CoreAnnotations.SentencesAnnotation.class).get(sentenceIndex));
    this.entityTokenSpan = Maybe.Just(contextSpan);
    this.document = Maybe.Just(contextDocument);
    this.sentenceIndex = Maybe.Just(sentenceIndex);
    this.properties = Maybe.Nothing();
    assert contextSpan.start() < sentence.get().get(CoreAnnotations.TokensAnnotation.class).size();
    assert contextSpan.end() <= sentence.get().get(CoreAnnotations.TokensAnnotation.class).size();
  }

  /**
   * Create a new entity context: the entity, context in which it's found.
   * @param entity The entity whose context we're representing.
   * @param contextDocument The document in which this entity occurs.
   * @param sentenceIndex The index of the sentence the entity mention occurs in.
   * @param contextSpan The span within the document in which this entity occurs.
   * @param properties Known properties of the entity.
   */
  public EntityContext(KBPEntity entity, Annotation contextDocument, int sentenceIndex, Span contextSpan, Collection<KBPSlotFill> properties) {
    this.entity = entity;
    this.sentence = Maybe.Just(contextDocument.get(CoreAnnotations.SentencesAnnotation.class).get(sentenceIndex));
    this.entityTokenSpan = Maybe.Just(contextSpan);
    this.document = Maybe.Just(contextDocument);
    this.sentenceIndex = Maybe.Just(sentenceIndex);
    this.properties = Maybe.Just(properties);
  }

  /**
   * Convert this object to its corresponding Protocol Buffer.
   */
  public KBPProtos.EntityContext toProto() {
    KBPProtos.EntityContext.Builder builder = KBPProtos.EntityContext.newBuilder();
    builder.setEntity(this.entity.toProto());
    for (CoreMap sentence : this.sentence) {
      ProtobufAnnotationSerializer serializer = new ProtobufAnnotationSerializer(false);
      builder.setSentence(serializer.toProto(sentence));
    }
    for (Span span : entityTokenSpan) { builder.setEntitySpan(KBPProtos.Span.newBuilder().setStart(span.start()).setEnd(span.end()).build()); }
    for (Collection<KBPSlotFill> props : this.properties) {
      for (KBPSlotFill prop : props) {
        builder.addProperties(prop.toProto());
      }
    }
    return builder.build();
  }

  /**
   * Read the entity context from its corresponding Protocol Buffer.
   */
  public static EntityContext fromProto(KBPProtos.EntityContext proto) {
    // Get relevant data
    KBPEntity entity = KBPNew.from(proto.getEntity()).KBPEntity();
    List<KBPSlotFill> properties = new ArrayList<>();
    for (KBPProtos.KBPSlotFill fill : proto.getPropertiesList()) {
      properties.add(KBPNew.from(fill).KBPSlotFill());
    }

    // Build entity context
    ProtobufAnnotationSerializer serializer = new ProtobufAnnotationSerializer();
    if (proto.hasSentence() && proto.hasEntitySpan() && properties.size() > 0) {
      return new EntityContext(entity, serializer.fromProto(proto.getSentence()),
          new Span(proto.getEntitySpan().getStart(), proto.getEntitySpan().getEnd()),
          properties);
    } else if (proto.hasSentence() && proto.hasEntitySpan()) {
      return new EntityContext(entity, serializer.fromProto(proto.getSentence()),
          new Span(proto.getEntitySpan().getStart(), proto.getEntitySpan().getEnd()));
    } else if (properties.size() > 0) {
      return new EntityContext(entity, properties);
    } else {
      return new EntityContext(entity);
    }
  }

  /**
   * Cache the tokenized form of the entity string.
   */
  private String[] tokens;

  /**
   * Get the tokenized entity.
   */
  public String[] tokens() {
    if (tokens == null) {
      tokens = entity.name.split("\\s+");
    }
    return tokens;
  }

  @Override
  public String toString() { return entity.toString(); }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (!(o instanceof EntityContext)) return false;
    EntityContext that = (EntityContext) o;

    if (this.sentence.isDefined() && that.sentence.isDefined()) {
      if (!this.sentence.get().get(CoreAnnotations.TextAnnotation.class).equals(that.sentence.get().get(CoreAnnotations.TextAnnotation.class))) {
        return false;
      }
    } else if (this.sentence.isDefined() || that.sentence.isDefined()) {
      return false;  // only one has a sentence defined
    }

    return entity.equals(that.entity) && entityTokenSpan.equals(that.entityTokenSpan);
  }

  @Override
  public int hashCode() {
    int result = entity.hashCode();
    result = 31 * result + entityTokenSpan.hashCode();
    result = 31 * result + properties.hashCode();
    return result;
  }
}
