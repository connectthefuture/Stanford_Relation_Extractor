package edu.stanford.nlp.kbp.common;

import java.io.Serializable;

/**
 * A class to encapsulate an abstract entity used in KBP.
 * This can either be an official entity
 * ({@link KBPOfficialEntity} -- a person or organization along with a query
 * id and the like) or a slot value (e.g., a date or location).
 *
 * @author Gabor Angeli
 */
public class KBPEntity implements Serializable, Comparable<KBPEntity> {
  private static final long serialVersionUID = 1L;

  public final String name;
  public final NERTag type;

  protected KBPEntity(String name, NERTag type) {
    assert name != null;
    this.name = name;
    assert type != null;
    this.type = type;
  }

  /**
   * Return whether this entity <b>could</b> be an official entity -- that is,
   * it is either a PERSON or ORGANIZATION (i.e., an official type)
   */
  public boolean isOfficial() {
    return type.isEntityType();
  }

  /**
   * Convert this object to its corresponding Protocol Buffer.
   */
  public KBPProtos.KBPEntity toProto() {
    KBPProtos.KBPEntity.Builder builder = KBPProtos.KBPEntity.newBuilder();
    return builder
        .setName(name)
        .setType(type.name)
        .build();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (!(o instanceof KBPEntity)) return false;
    KBPEntity kbpEntity = (KBPEntity) o;
    return name.equals(kbpEntity.name) && type == kbpEntity.type;
  }

  private volatile int hashCodeCache = 0;
  @Override
  public int hashCode() {
    if (hashCodeCache == 0) {
      hashCodeCache = name.toLowerCase().hashCode();
    }
    return hashCodeCache;
  }

  @Override
  public String toString() { return type + ":" + name; }

  /**
   * This is only for stable ordering. There's no meaningful natural ordering of entities.
   */
  @Override
  public int compareTo(KBPEntity o) {
    // Try to be as random as possible, to avoid unintentional bias from alphabetization
    int hashA = this.hashCode();
    int hashB = o.hashCode();
    if (hashA != hashB) {
      return hashA - hashB;
    } else {
      return new StringBuilder().append(this.type).append(":").append(this.name).reverse().toString().compareTo(
        new StringBuilder().append(o.type).append(":").append(o.name).reverse().toString());
    }
  }
}
