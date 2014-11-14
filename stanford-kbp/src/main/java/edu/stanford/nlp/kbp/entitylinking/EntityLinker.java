package edu.stanford.nlp.kbp.entitylinking;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.kbp.common.*;
import java.util.function.Function;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.logging.Redwood;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.*;

/**
 * KBP Slotfilling's view into Entity Linking.
 *
 * @author Gabor Angeli
 */
public abstract class EntityLinker implements Function<Pair<EntityContext, EntityContext>, Boolean> {
  /**
   * A set of standard corporate suffixes, to be used in determining if two companies are the same modulo
   * a standard corporate marker.
   */
  public static final Set<String> CORPORATE_SUFFIXES = Collections.unmodifiableSet(new HashSet<String>() {{
    // From http://en.wikipedia.org/wiki/Types_of_companies#United_Kingdom
    add("cic"); add("cio"); add("general partnership"); add("llp"); add("llp."); add("limited liability partnership");
    add("lp"); add("lp."); add("limited partnership"); add("ltd"); add("ltd."); add("plc"); add("plc.");
    add("private company limited by guarantee"); add("unlimited company"); add("sole proprietorship");
    add("sole trader");
    // From http://en.wikipedia.org/wiki/Types_of_companies#United_States
    add("na"); add("nt&sa"); add("federal credit union"); add("federal savings bank"); add("lllp"); add("lllp.");
    add("llc"); add("llc."); add("lc"); add("lc."); add("ltd"); add("ltd."); add("co"); add("co.");
    add("pllc"); add("pllc."); add("corp"); add("corp."); add("inc"); add("inc.");
    add("pc"); add("p.c."); add("dba");
    // From state requirements section
    add("corporation"); add("incorporated"); add("limited"); add("association"); add("company"); add("clib");
    add("syndicate"); add("institute"); add("fund"); add("foundation"); add("club"); add("partners");
  }});

  /**
   * A short set of determiners.
   */
  public static final Set<String> DETERMINERS = Collections.unmodifiableSet(new HashSet<String>() {{
    add("the"); add("The"); add("a"); add("A");
  }});

  /** A map from male names to their canonical form; e.g., Ron -&gt; Aaron */
  protected final Map<String, String> maleNamesLowerCase;
  /** A map from female names to their canonical form; e.g., Abby -&gt; Abigail */
  protected final Map<String, String> femaleNamesLowerCase;

  public EntityLinker() {
    // Load names
    maleNamesLowerCase = Collections.unmodifiableMap(readNicknames(Props.ENTITYLINKING_MALENAMES.getPath()));
    femaleNamesLowerCase = Collections.unmodifiableMap(readNicknames(Props.ENTITYLINKING_FEMALENAMES.getPath()));
  }

  /**
   * The top level entry point into the entity linker; equivalent to {@link EntityLinker#apply(edu.stanford.nlp.util.Pair)}.
   * Returns true if the two entity contexts refer to the same entity, and false otherwise.
   * It will try to first link the two contexts and check if they link to the same entity, and then determine
   * if the two entities are the same without having to explicitly link them.
   *
   * @param a One of the entity contexts.
   * @param b The other entity context.
   * @return True if these two contexts refer to the same entity.
   */
  public boolean sameEntity(EntityContext a, EntityContext b) {
    // Try conventional entity linking
    for (String id1 : link(a)) {
      //noinspection LoopStatementThatDoesntLoop
      for (String id2 : link(b)) {
        return id1.equals(id2);
      }
    }

    // Check some hard constraints
    // Enforce same type
    if (a.entity.type != b.entity.type) { return false; }
    // Check for acronym match
    if (AcronymMatcher.isAcronym(a.tokens(), b.tokens())) { return true; }

    // Backoff to the implementing model
    return sameEntityWithoutLinking(a, b);
  }

  @Override
  public Boolean apply(Pair<EntityContext, EntityContext> in) {
    return sameEntity(in.first, in.second);
  }

  /**
   * Return a unique id for this entity, if one is available.
   * @param context The entity to link, along with all known context.
   * @return The id of the entity, if one could be found.
   */
  public abstract Maybe<String> link(EntityContext context);

  /**
   * If the entity could not be linked, try to determine if two entities are the same anyways.
   * @param entityOne The first entity, with its context.
   * @param entityTwo The second entity, with its context.
   * @return True if the two entities are the same entity in reality.
   */
  protected abstract boolean sameEntityWithoutLinking(EntityContext entityOne, EntityContext entityTwo);


  protected abstract void printJustification(EntityContext entityOne, EntityContext entityTwo);

  /**
   * A utility to strip out corporate titles (e.g., "corp.", "incorporated", etc.)
   * @param input The string to strip titles from
   * @return A string without these titles, or the input string if not such titles exist.
   */
  protected String stripCorporateTitles(String input) {
    for (String suffix : CORPORATE_SUFFIXES) {
      if (input.toLowerCase().endsWith(suffix)) {
        return input.substring(0, input.length() - suffix.length()).trim();
      }
    }
    return input;
  }

  /**
   * A utility to strip out leading determiners ("a", "the", etc.)
   * @param input The string to strip determiners from.
   * @return A string without the leading determiners, or the original string if no determiners were found.
   */
  protected String stripDeterminers(String input) {
    for (String determiner : DETERMINERS) {
      if (input.startsWith(determiner)) { input = input.substring(determiner.length()).trim(); }
    }
    return input.trim();
  }

  private Map<String, String> readNicknames(String classpathOrFile) {
    BufferedReader reader = null;
    try {
      Map<String, String> names = new HashMap<String, String>();
      reader = IOUtils.getBufferedReaderFromClasspathOrFileSystem(classpathOrFile);
      String line;
      while ((line = reader.readLine()) != null) {
        String canonicalName = null;
        for (String chunk : line.split("\\t")) {
          for (String name : chunk.split(",")) {
            if (canonicalName == null) { canonicalName = name; }
            names.put(name.toLowerCase(), canonicalName);
          }
        }
      }
      return names;
    } catch (IOException e) {
      Redwood.Util.err(e);
      return new HashMap<>();
    } finally {
      if (reader != null) {
        try {
          reader.close();
        } catch (IOException ignored) { }
      }
    }
  }



  /**
   * A very dumb entity linker that checks for hard constraints, in addition to anything checked
   * by the {@link EntityLinker} base class.
   */
  public static class HardConstraintsEntityLinker extends EntityLinker {
    @Override
    public Maybe<String> link(EntityContext context) {
      return Maybe.Nothing();
    }
    @Override
    protected boolean sameEntityWithoutLinking(EntityContext entityOne, EntityContext entityTwo) {
      return entityOne.entity.name.equals(entityTwo.entity.name);
    }

    @Override
    protected void printJustification(EntityContext entityOne, EntityContext entityTwo) {

    }
  }

  /**
   * The hacky baseline Gabor tuned for KBP2013.
   * Please do better than me!
   */
  public static class GaborsHackyBaseline extends EntityLinker {
    @Override
    public Maybe<String> link(EntityContext context) {
      return Maybe.Nothing();
    }

    @SuppressWarnings({"RedundantIfStatement", "StringEquality"})
    @Override
    protected boolean sameEntityWithoutLinking(EntityContext entityOne, EntityContext entityTwo) {
      NERTag type = entityOne.entity.type;
      // Nicknames
      if (type == NERTag.PERSON && entityOne.tokens().length == 2 && entityTwo.tokens().length == 2 &&
          entityOne.tokens()[entityOne.tokens().length - 1].toLowerCase().equals(entityTwo.tokens()[entityTwo.tokens().length - 1].toLowerCase())) {
        // case: last names match
        String firstNameOne = entityOne.tokens()[0].toLowerCase();
        String firstNameTwo = entityTwo.tokens()[0].toLowerCase();
        //noinspection StringEquality  // Safe, as the Strings are canonicalized; != allows for more elegant handling of null,StringEquality
        if (maleNamesLowerCase.get(firstNameOne) != maleNamesLowerCase.get(firstNameTwo) ||
            femaleNamesLowerCase.get(firstNameOne) != femaleNamesLowerCase.get(firstNameTwo)) {
          // First names are not nicknames of each other -- they're likely different people.
          return false;
        }
      }


      // Disallow middle name match
      String[] shorterTokens = entityOne.tokens().length < entityTwo.tokens().length ? entityOne.tokens() : entityTwo.tokens();
      String[] longerTokens = entityOne.tokens().length < entityTwo.tokens().length ? entityTwo.tokens() : entityOne.tokens();
      if (type == NERTag.PERSON && longerTokens.length == 3 && shorterTokens.length == 2) {
        if (shorterTokens[0].equals(longerTokens[0]) && shorterTokens[1].equals(longerTokens[2])) {
          return false;  // middle name match
        }
      }

      // Proper match score
      double matchScore = Math.max(
          approximateEntityMatchScore(entityOne.entity.name, entityTwo.entity.name),
          approximateEntityMatchScore(entityTwo.entity.name, entityOne.entity.name));


      // Some simple cases
      if( matchScore == 1.0 ) { return true; }
      if( matchScore < 0.34 ) { return false; }
      if (type == NERTag.PERSON && matchScore > 0.49) {
        // Both entities are more than one character
        if (Math.min(entityOne.tokens().length, entityTwo.tokens().length) > 1) { return true; }
        // Last names match
        if (entityOne.tokens().length == 1 && entityTwo.tokens().length > 1 && entityTwo.tokens()[entityTwo.tokens().length - 1].equalsIgnoreCase(entityOne.tokens()[0])) {
          return true;
        }
        if (entityTwo.tokens().length == 1 && entityOne.tokens().length > 1 && entityOne.tokens()[entityOne.tokens().length - 1].equalsIgnoreCase(entityTwo.tokens()[0])) {
          return true;
        }
      }
      if (type == NERTag.ORGANIZATION && matchScore > 0.79) { return true; }

      // See if we can use properties
      for (Collection<KBPSlotFill> fillsOne : entityOne.properties) {
        for (Collection<KBPSlotFill> fillsTwo : entityTwo.properties) {
          Set<Pair<String, String>> propsOne = new HashSet<Pair<String, String>>();
          for (KBPSlotFill fill : fillsOne) { propsOne.add(Pair.makePair(fill.key.relationName, fill.key.slotValue)); }
          Set<Pair<String, String>> propsTwo = new HashSet<Pair<String, String>>();
          for (KBPSlotFill fill : fillsTwo) { propsTwo.add(Pair.makePair(fill.key.relationName, fill.key.slotValue)); }
          int overlap = CollectionUtils.allOverlaps(propsOne, propsTwo).size();
          int minSize = Math.min(propsOne.size(), propsTwo.size());
          if (minSize > 1 && overlap >= minSize / 2) { return true; }
        }
      }

      // Check for subsidiary-like relations
      if (type == NERTag.ORGANIZATION) {
        String gloss1 = entityOne.entity.name.toLowerCase();
        String gloss2 = entityTwo.entity.name.toLowerCase();
        if (gloss1.contains("chapter") || gloss2.contains("chapter")) { return false; }
        if (gloss1.contains("department") || gloss2.contains("department")) { return false; }
        if (gloss1.contains("division") || gloss2.contains("division")) { return false; }
        if (gloss1.contains("section") || gloss2.contains("section")) { return false; }
        if (gloss1.contains("branch") || gloss2.contains("section")) { return false; }
        if (gloss1.contains("office") || gloss2.contains("office")) { return false; }
      }

      // Default to false
      return false;
    }

    @Override
    protected void printJustification(EntityContext entityOne, EntityContext entityTwo) {

    }

    private static boolean nearExactEntityMatch( String higherGloss, String lowerGloss ) {
      // case: slots have same relation, and that relation isn't an alternate name
      // Filter case sensitive match
      if (higherGloss.equalsIgnoreCase(lowerGloss)) { return true; }
      // Ignore certain characters
      else if (Utils.noSpecialChars(higherGloss).equalsIgnoreCase(Utils.noSpecialChars(lowerGloss))) { return true; }
      return false;
    }

    /**
     * Approximately check if two entities are equivalent.
     * Taken largely from
     * edu.stanford.nlp.kbp.slotfilling.evaluate,HeuristicSlotfillPostProcessors.NoDuplicatesApproximate;
     */
    protected double approximateEntityMatchScore( String higherGloss, String lowerGloss) {
      if( nearExactEntityMatch(higherGloss, lowerGloss) ) return 1.0;

      // Partial names
      String[] higherToks = stripCorporateTitles(higherGloss).split("\\s+");
      String[] lowerToks = stripCorporateTitles(lowerGloss).split("\\s+");
      // Case: acronyms of each other
      if (AcronymMatcher.isAcronym(higherToks, lowerToks)) { return 1.0; }

      int match = 0;
      // Get number of matching tokens between the two slot fills
      boolean[] matchedHigherToks = new boolean[higherToks.length];
      boolean[] matchedLowerToks = new boolean[lowerToks.length];
      for (int h = 0; h < higherToks.length; ++h) {
        if (matchedHigherToks[h]) { continue; }
        String higherTok = higherToks[h];
        String higherTokNoSpecialChars = Utils.noSpecialChars(higherTok);
        boolean doesMatch = false;
        for (int l = 0; l < lowerToks.length; ++l) {
          if (matchedLowerToks[l]) { continue; }
          String lowerTok = lowerToks[l];
          String lowerTokNoSpecialCars = Utils.noSpecialChars(lowerTok);
          int minLength = Math.min(lowerTokNoSpecialCars.length(), higherTokNoSpecialChars.length());
          if (higherTokNoSpecialChars.equalsIgnoreCase(lowerTokNoSpecialCars) ||  // equal
              (minLength > 5 && (higherTokNoSpecialChars.endsWith(lowerTokNoSpecialCars) || higherTokNoSpecialChars.startsWith(lowerTokNoSpecialCars))) ||  // substring
              (minLength > 5 && (lowerTokNoSpecialCars.endsWith(higherTokNoSpecialChars) || lowerTokNoSpecialCars.startsWith(higherTokNoSpecialChars))) ||  // substring the other way
              (minLength > 5 && Utils.levenshteinDistance(lowerTokNoSpecialCars, higherTokNoSpecialChars) <= 1)  // edit distance <= 1
              ) {
            doesMatch = true;  // a loose metric of "same token"
            matchedHigherToks[h] = true;
            matchedLowerToks[l] = true;
          }
        }
        if (doesMatch) { match += 1; }
      }

      return (double) match / ((double) Math.max(higherToks.length, lowerToks.length));
    }
  }

  /**
   * Please help me get rid of this!
   */
  public static class GaborsHackyDuplicateDetector extends GaborsHackyBaseline {
    @SuppressWarnings({"RedundantIfStatement", "StringEquality"})
    @Override
    protected boolean sameEntityWithoutLinking(EntityContext entityOne, EntityContext entityTwo) {
      if (entityOne.entity.name.contains(entityTwo.entity.name) || entityTwo.entity.name.contains(entityOne.entity.name)) {
        return true;
      }
      NERTag type = entityOne.entity.type;
      // Nicknames
      if (type == NERTag.PERSON && entityOne.tokens().length == 2 && entityTwo.tokens().length == 2 &&
          entityOne.tokens()[entityOne.tokens().length - 1].toLowerCase().equals(entityTwo.tokens()[entityTwo.tokens().length - 1].toLowerCase())) {
        // case: last names match
        String firstNameOne = entityOne.tokens()[0].toLowerCase();
        String firstNameTwo = entityTwo.tokens()[0].toLowerCase();
        //noinspection StringEquality  // Safe, as the Strings are canonicalized; != allows for more elegant handling of null,StringEquality
        if (maleNamesLowerCase.get(firstNameOne) != maleNamesLowerCase.get(firstNameTwo) ||
            femaleNamesLowerCase.get(firstNameOne) != femaleNamesLowerCase.get(firstNameTwo)) {
          // First names are not nicknames of each other -- they're likely different people.
          return false;
        }
      }
      // Proper match score
      double matchScore = Math.max(
          approximateEntityMatchScore(entityOne.entity.name, entityTwo.entity.name),
          approximateEntityMatchScore(entityTwo.entity.name, entityOne.entity.name));


      // Some simple cases
      if( matchScore == 1.0 ) { return true; }
      if( matchScore < 0.34 ) { return false; }
      if (type == NERTag.PERSON && matchScore > 0.49) { return true; }
      if (type == NERTag.ORGANIZATION && matchScore > 0.79) { return true; }

      // See if we can use properties
      for (Collection<KBPSlotFill> fillsOne : entityOne.properties) {
        for (Collection<KBPSlotFill> fillsTwo : entityTwo.properties) {
          Set<Pair<String, String>> propsOne = new HashSet<Pair<String, String>>();
          for (KBPSlotFill fill : fillsOne) { propsOne.add(Pair.makePair(fill.key.relationName, fill.key.slotValue)); }
          Set<Pair<String, String>> propsTwo = new HashSet<Pair<String, String>>();
          for (KBPSlotFill fill : fillsTwo) { propsTwo.add(Pair.makePair(fill.key.relationName, fill.key.slotValue)); }
          int overlap = CollectionUtils.allOverlaps(propsOne, propsTwo).size();
          int minSize = Math.min(propsOne.size(), propsTwo.size());
          if (minSize > 1 && overlap >= minSize / 2) { return true; }
        }
      }

      // Check for subsidiary-like relations
      if (type == NERTag.ORGANIZATION) {
        String gloss1 = entityOne.entity.name.toLowerCase();
        String gloss2 = entityTwo.entity.name.toLowerCase();
        if (gloss1.contains("chapter") || gloss2.contains("chapter")) { return false; }
        if (gloss1.contains("department") || gloss2.contains("department")) { return false; }
        if (gloss1.contains("division") || gloss2.contains("division")) { return false; }
        if (gloss1.contains("section") || gloss2.contains("section")) { return false; }
        if (gloss1.contains("branch") || gloss2.contains("section")) { return false; }
        if (gloss1.contains("office") || gloss2.contains("office")) { return false; }
      }

      // Default to false
      return true;

    }
  }



  /**
   * The hacky baseline Gabor made for mining inferential paths, comparing certain properties of
   * the two entities to link.
   */
  public static class GaborsHighPrecisionBaseline extends EntityLinker {
    @Override
    public Maybe<String> link(EntityContext context) { return Maybe.Nothing(); }

    @Override
    protected boolean sameEntityWithoutLinking(EntityContext entityOne, EntityContext entityTwo) {
      // Check name equality
      if (entityOne.entity.name.equals(entityTwo.entity.name)) { return true; }
      for (Collection<KBPSlotFill> props1 : entityOne.properties) {
        // Check alternate names
        for (KBPSlotFill fill1 : props1) {
          if (fill1.key.hasKBPRelation() && fill1.key.kbpRelation().isAlternateName() && fill1.key.slotValue.equals(entityTwo.entity.name)) { return true; }
          if (fill1.key.slotValue.equals(entityOne.entity.name)) { return false; }  // should not have relation with ones self
        }
        //noinspection LoopStatementThatDoesntLoop
        for (Collection<KBPSlotFill> props2 : entityTwo.properties) {
          // Check alternate names
          for (KBPSlotFill fill2 : props1) {
            if (fill2.key.hasKBPRelation() && fill2.key.kbpRelation().isAlternateName() && fill2.key.slotValue.equals(entityOne.entity.name)) { return true; }
            if (fill2.key.slotValue.equals(entityTwo.entity.name)) { return false; }  // should not have relation with ones self
          }
          // Check relation overlap
          Set<String> overlap = commonProperties(props1, props2);
          return overlap.contains(RelationType.PER_DATE_OF_BIRTH.canonicalName) ||
              overlap.contains(RelationType.PER_DATE_OF_DEATH.canonicalName) ||
              overlap.contains(RelationType.ORG_FOUNDED.canonicalName) ||
              (overlap.contains(RelationType.PER_TITLE.canonicalName) && overlap.size() > 1) ||
              (overlap.contains(RelationType.PER_CITY_OF_BIRTH.canonicalName) && overlap.size() > 1) ||
              (overlap.contains(RelationType.PER_CITY_OF_DEATH.canonicalName) && overlap.size() > 1) ||
              (overlap.contains(RelationType.PER_COUNTRY_OF_BIRTH.canonicalName) && overlap.size() > 1) ||
              (overlap.contains(RelationType.PER_COUNTRY_OF_DEATH.canonicalName) && overlap.size() > 1) ||
              (overlap.contains(RelationType.PER_STATE_OR_PROVINCES_OF_BIRTH.canonicalName) && overlap.size() > 1) ||
              (overlap.contains(RelationType.PER_STATE_OR_PROVINCES_OF_DEATH.canonicalName) && overlap.size() > 1) ||
              (overlap.size() == 2 && (props1.size() == 3 || props2.size() == 3)) ||
              overlap.size() > Math.max(2, Math.max(props1.size(), props2.size()) / 2);
        }
      }
      return false;
    }

    @Override
    protected void printJustification(EntityContext entityOne, EntityContext entityTwo) {

    }

    private Set<String> commonProperties(Collection<KBPSlotFill> a, Collection<KBPSlotFill> b) {
      Set<String> overlap = new HashSet<String>();
      for (KBPSlotFill fillA : a) {
        for (KBPSlotFill fillB : b) {
          if (fillA.key.relationName.equals(fillB.key.relationName) &&
              fillA.key.slotValue.trim().equals(fillB.key.slotValue.trim())) {
            overlap.add(fillA.key.relationName);
          }
        }
      }
      return overlap;
    }
  }
}
