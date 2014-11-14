
package edu.stanford.nlp.kbp.slotfilling.ir.index;
 
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.IntField;

import java.util.HashMap;
import java.util.Map;


/**
 * This enum contains the different fields we keep in the Lucene
 * repository.  When text names are needed, such as for field names,
 * the strings are the enum's .toString() method.
 */
public enum KBPField {
  DATETIME        ("datetime",       KBPTypes.NOT_ANALYZED),
  DOCID           ("docid",          KBPTypes.NOT_ANALYZED),
  DOCTYPE         ("doctype",        KBPTypes.NOT_ANALYZED),
  DOCSOURCETYPE   ("docsourcetype",  KBPTypes.NOT_ANALYZED),
  TITLE           ("title",          KBPTypes.ANALYZED),
  TEXT            ("text",           KBPTypes.ANALYZED),
  SCHOOL          ("school",         KBPTypes.ANALYZED),
  WIKITITLE       ("title",          KBPTypes.NOT_ANALYZED), 
  WIKICONTENT     ("content",        KBPTypes.ANALYZED),
  COREMAP         ("coremap",        KBPTypes.NOT_INDEXED),
  COREMAP_VERSION ("coremapVersion", IntField.TYPE_STORED),
  COREMAP_FILE    ("coremapFile",    KBPTypes.NOT_INDEXED),  // stores location of the coremap on disk

  // Text tokenized by stanford corenlp
  // TODO: Fix types
  TEXT_WORD        ("text_word",      KBPTypes.ANALYZED),
  TEXT_WORD_NORM   ("text_word_norm", KBPTypes.ANALYZED_NOT_STORED),
  TEXT_ANNOTATED   ("text_annotated", KBPTypes.ANALYZED),
  TEXT_LEMMA       ("text_lemma",     KBPTypes.ANALYZED_NO_POSITION),
  TEXT_POS         ("text_pos",       KBPTypes.ANALYZED_NO_POSITION),
  TEXT_NER         ("text_ner",       KBPTypes.ANALYZED_NO_POSITION),
  TEXT_NER_NORM    ("text_ner_norm",  KBPTypes.ANALYZED_NO_POSITION),

  // Pre-analyzed text - experimental
  TEXT_PRE        ("text_pre", KBPTypes.NOT_ANALYZED),

  // Additional fields that we index on
  LOCATION ("location", KBPTypes.NOT_ANALYZED),
  AUTHORS  ("authors",  KBPTypes.NOT_ANALYZED),
  SUBJECTS ("subjects", KBPTypes.NOT_ANALYZED),

  ENTITIES ("entities", KBPTypes.NOT_ANALYZED),
  ENTITIES_NAME ("entities_name", KBPTypes.NOT_ANALYZED),
  ENTITIES_TEXT ("entities_text", KBPTypes.NOT_ANALYZED),
  ENTITIES_TYPE ("entities_type", KBPTypes.NOT_ANALYZED);

  public static class KBPTypes {
  
    /* Indexed, tokenized, stored. */
    public static final FieldType ANALYZED = new FieldType();
    public static final FieldType ANALYZED_NO_POSITION = new FieldType();
    /* Indexed, tokenized, not stored. */
    public static final FieldType ANALYZED_NOT_STORED = new FieldType();

    /* Indexed, not tokenized, stored. */
    public static final FieldType NOT_ANALYZED = new FieldType();
    /* not Indexed, not tokenized, stored. */
    public static final FieldType NOT_INDEXED = new FieldType();

    static {
      ANALYZED_NOT_STORED.setIndexed(true);
      ANALYZED_NOT_STORED.setTokenized(true);
      ANALYZED_NOT_STORED.setStored(false);
      ANALYZED_NOT_STORED.setStoreTermVectors(true);
      ANALYZED_NOT_STORED.setStoreTermVectorPositions(true);
      ANALYZED_NOT_STORED.freeze();

      ANALYZED.setIndexed(true);
      ANALYZED.setTokenized(true);
      ANALYZED.setStored(true);
      ANALYZED.setStoreTermVectors(true);
      ANALYZED.setStoreTermVectorPositions(true);
      ANALYZED.freeze();
      
      ANALYZED_NO_POSITION.setIndexed(true);
      ANALYZED_NO_POSITION.setTokenized(true);
      ANALYZED_NO_POSITION.setStoreTermVectors(true);
      ANALYZED_NO_POSITION.setStoreTermVectorPositions(false);
      ANALYZED_NO_POSITION.freeze();

      NOT_ANALYZED.setIndexed(true);
      NOT_ANALYZED.setTokenized(false);
      NOT_ANALYZED.setStored(true);
      NOT_ANALYZED.setStoreTermVectors(false);
      NOT_ANALYZED.setStoreTermVectorPositions(false);
      NOT_ANALYZED.freeze();

      NOT_INDEXED.setIndexed(false);
      NOT_INDEXED.setTokenized(false);
      NOT_INDEXED.setStored(true);
      NOT_INDEXED.setStoreTermVectors(false);
      NOT_INDEXED.setStoreTermVectorPositions(false);
      NOT_INDEXED.freeze();
    }
  }

  private static Map<String,KBPField> fieldsByName = new HashMap<String,KBPField>();
  static {
    for (KBPField v: KBPField.values()) {
      fieldsByName.put(v.fieldName, v);
    }
  }

  public static KBPField lookupField(String fieldName) {
    return fieldsByName.get(fieldName);
  }

  private final String fieldName;
  private final FieldType indexingStrategy;
  KBPField(String fieldName, FieldType indexingStrategy) {
    this.fieldName = fieldName;
    this.indexingStrategy = indexingStrategy;
  }

  public String fieldName() {
    return fieldName;
  }
  
  public FieldType indexingStrategy() {
    return indexingStrategy;
  }
}

