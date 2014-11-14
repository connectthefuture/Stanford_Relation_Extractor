package edu.stanford.nlp.kbp.common;

/**
* An expanded set of NER tags
*
* @author Gabor Angeli
*/
public enum NERTag {
  // ENUM_NAME        NAME           SHORT_NAME  IS_REGEXNER_TYPE
  CAUSE_OF_DEATH    ("CAUSE_OF_DEATH",    "COD", true), // note: these names must be upper case
  CITY              ("CITY",              "CIT", true), //       furthermore, DO NOT change the short names, or else serialization may break
  COUNTRY           ("COUNTRY",           "CRY", true),
  CRIMINAL_CHARGE   ("CRIMINAL_CHARGE",   "CC",  true),
  DATE              ("DATE",              "DT",  false),
  IDEOLOGY          ("IDEOLOGY",          "IDY", true),
  LOCATION          ("LOCATION",          "LOC", false),
  MISC              ("MISC",              "MSC", false),
  MODIFIER          ("MODIFIER",          "MOD", false),
  NATIONALITY       ("NATIONALITY",       "NAT", true),
  NUMBER            ("NUMBER",            "NUM", false),
  ORGANIZATION      ("ORGANIZATION",      "ORG", false),
  PERSON            ("PERSON",            "PER", false),
  RELIGION          ("RELIGION",          "REL", true),
  STATE_OR_PROVINCE ("STATE_OR_PROVINCE", "ST",  true),
  TITLE             ("TITLE",             "TIT", true),
  URL               ("URL",               "URL", true),
  DURATION          ("DURATION",          "DUR", false),
  ;

  /** The full name of this NER tag, as would come out of our NER or RegexNER system */
  public final String name;
  /** A short name for this NER tag, intended for compact serialization */
  public final String shortName;
  /** If true, this NER tag is not in the standard NER set, but is annotated via RegexNER */
  public final boolean isRegexNERType;
  NERTag(String name, String shortName, boolean isRegexNERType){ this.name = name; this.shortName = shortName; this.isRegexNERType = isRegexNERType; }

  /** Find the slot for a given name */
  public static Maybe<NERTag> fromString(String name) {
    // Early termination
    if (name == null || name.equals("")) { return Maybe.Nothing(); }
    // Cycle known NER tags
    name = name.toUpperCase();
    for (NERTag slot : NERTag.values()) {
      if (slot.name.equals(name)) return Maybe.Just(slot);
    }
    for (NERTag slot : NERTag.values()) {
      if (slot.shortName.equals(name)) return Maybe.Just(slot);
    }
    // Some quick fixes
    return Maybe.Nothing();
  }

  /** Find the slot for a given name */
  public static Maybe<NERTag> fromShortName(String name) {
    // Early termination
    if (name == null) { return Maybe.Nothing(); }
    // Cycle known NER tags
    name = name.toUpperCase();
    for (NERTag slot : NERTag.values()) {
      if (slot.shortName.startsWith(name)) return Maybe.Just(slot);
    }
    // Some quick fixes
    return Maybe.Nothing();
  }


  public String toXMLRepresentation() {
    switch (this) {
      case PERSON: return "PER";
      case ORGANIZATION: return "ORG";
      default: return name;
    }
  }

  public static Maybe<NERTag> fromRelation(String relation) {
    assert( relation.length() >= 3 );
    relation = relation.toLowerCase();
    if (relation.startsWith("per:")) return Maybe.Just(PERSON);
    else if (relation.startsWith("org:")) return Maybe.Just(ORGANIZATION);
    else return Maybe.Nothing();
  }

  public boolean isEntityType() {
    return this.equals(PERSON) || this.equals(ORGANIZATION);
  }

  public boolean isGeographic() {
    return this.equals(LOCATION) || this.equals(COUNTRY) || this.equals(STATE_OR_PROVINCE) || this.equals(CITY);
  }
}
