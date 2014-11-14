package edu.stanford.nlp.kbp.common;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.util.ArrayUtils;

import java.util.Comparator;
import java.util.HashSet;
import java.util.Set;

/**
 * <p>An official KBP query entity -- that is, the first argument to a triple (entity, relation, slotValue).</p>
 *
 * <p>
 *   Important Implementation Notes:
 * </p>
 *   <ul>
 *     <li><b>Equality:</b> To be equal to another KBPEntity, it must have the same name and type.
 *                          To be equal to another KBPOfficialEntity, it must have the same ID.
 *                          BUT: for hashing purposes, it must also have the same name, or else the hash code will not bucket
 *                               the entity correctly.</li>
 *   </ul>
 */
public class KBPOfficialEntity extends KBPEntity {
  private static final long serialVersionUID = 2L;

  public final Maybe<String> id;
  public final Maybe<String> queryId;
  public final Maybe<Set<RelationType>> ignoredSlots;
  public final Maybe<EntityContext> representativeContext;

  public Maybe<String> representativeDocumentId() {
    if (representativeContext.isDefined() && representativeContext.get().document.isDefined()) {
      return Maybe.Just(representativeContext.get().document.get().get(CoreAnnotations.DocIDAnnotation.class));
    } else {
      return Maybe.Nothing();
    }
  }

  protected KBPOfficialEntity(String name,
                           NERTag type,
                           Maybe<String> id,
                           Maybe<String> queryId,
                           Maybe<Set<RelationType>> ignoredSlots,
                           Maybe<EntityContext> representativeContext) {
    super(name, type);
    if (!type.isEntityType()) { throw new IllegalArgumentException("Invalid entity type for official entity: " + type); }
    assert id != null;
    assert id.getOrElse("") != null;
    this.id = id;
    assert queryId != null;
    assert queryId.getOrElse("") != null;
    this.queryId = queryId;
    assert ignoredSlots != null;
    assert ignoredSlots.getOrElse(new HashSet<RelationType>()) != null;
    this.ignoredSlots = ignoredSlots;
    assert representativeContext != null;
    assert queryId.getOrElse("") != null;
    this.representativeContext = representativeContext;
  }

  /**
   * Convert this object to its corresponding Protocol Buffer.
   */
  @Override
  public KBPProtos.KBPEntity toProto() {
    KBPProtos.KBPEntity.Builder builder = KBPProtos.KBPEntity.newBuilder(super.toProto());
    for (String x : id) { builder.setId(x); }
    for (String x : queryId) { builder.setQueryId(x); }
    for (EntityContext x : representativeContext) { builder.setRepresentativeDocument(x.toProto()); }
    for (Set<RelationType> relations : ignoredSlots) {
      for (RelationType rel : relations) {
        builder.addIgnoredSlots(rel.canonicalName);
      }
    }
    return builder.build();
  }

  @Override
  public int hashCode() {
    return name.toLowerCase().hashCode();
  }

  @Override
  public boolean equals(Object obj) {
    if(obj instanceof KBPOfficialEntity){
      KBPOfficialEntity em = (KBPOfficialEntity) obj;
      // NOTE (arun): This used to assert that id != null earlier. I'm relaxing the constraint
      // to first check on id then others.
      if (id.isDefined() && em.id.isDefined()) {
        // Note: names must match if id's match; this is a safety check to make sure hashing works correctly
        if (Utils.assertionsEnabled() && id.get().equals(em.id.get()) && !name.toLowerCase().equals(em.name.toLowerCase())) {
          throw new AssertionError("Official entity with same id has different name: " + name + " vs " + em.name);
        }
        return id.get().equals(em.id.get());
      } else {
        return type == em.type && name.equals(em.name);
      }
    } else if (obj instanceof KBPEntity) {
      KBPEntity em = (KBPEntity) obj;
      return type == em.type && name.equals(em.name);
    }
    return false;
  }

  @Override
  public String toString() {
    String s = type + ":" + name;
    s += " (" + id.getOrElse("<no id>") + "," + queryId.getOrElse("<no query id>") + ")";
    return s;
  }

  public static class QueryIdSorter implements Comparator<KBPOfficialEntity> {
    public int compare(KBPOfficialEntity first, KBPOfficialEntity second) {
      // queryId should be unique, so there shouldn't be a need for a
      // fallback comparison
      return first.queryId.orCrash().compareTo(second.queryId.orCrash());
    }
  }
  
  /**
   * Sort first alphabetically by entity name, then by query ID, then by entity ID, then by type.
   */
  @SuppressWarnings("UnusedDeclaration")
  public static class AlphabeticSorter implements Comparator<KBPOfficialEntity> {
    public int compare(KBPOfficialEntity first, KBPOfficialEntity second) {
      return ArrayUtils.compareArrays(extractFields(first), extractFields(second));
    }

    private String[] extractFields(KBPOfficialEntity entity) {
      String queryId = entity.queryId.getOrElse("(null)");
      String id = entity.id.getOrElse("(null)");
      return new String[] { entity.name, queryId, id, entity.type.toString() };
    }
  }
}
