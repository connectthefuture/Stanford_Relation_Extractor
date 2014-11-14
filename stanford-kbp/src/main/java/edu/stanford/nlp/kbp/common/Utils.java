package edu.stanford.nlp.kbp.common;

import edu.stanford.nlp.ie.NumberNormalizer;
import edu.stanford.nlp.ie.machinereading.structure.EntityMention;
import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.kbp.slotfilling.evaluate.EntityGraph;
import edu.stanford.nlp.kbp.slotfilling.evaluate.KBPScore;
import edu.stanford.nlp.kbp.slotfilling.evaluate.WorldKnowledgePostProcessor;
import edu.stanford.nlp.ling.CoreAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.*;
import edu.stanford.nlp.util.logging.Color;
import edu.stanford.nlp.util.logging.Redwood;

import java.lang.reflect.Array;
import java.text.DecimalFormat;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.regex.Pattern;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

@SuppressWarnings("UnusedDeclaration")
public class Utils {


  private Utils() {} // static methods

  public static WorldKnowledgePostProcessor geography() { return WorldKnowledgePostProcessor.singleton(Props.TEST_CONSISTENCY_WORLDKNOWLEDGE_DIR); }


  public static String makeNERTag(NERTag et) {
    return ("ENT:" + et.name);
  }

  public static Maybe<NERTag> getNERTag(EntityMention e) {
    String typeString = e.getType();
    if( typeString != null && typeString.length() > 0 ) {
      switch (typeString) {
        case "ENT:PERSON":
          return Maybe.Just(NERTag.PERSON);
        case "ENT:ORGANIZATION":
          return Maybe.Just(NERTag.ORGANIZATION);
        default:
          return NERTag.fromString(typeString);
      }
    } else {
      return Maybe.Nothing();
    }
  }


  public static Maybe<String> getKbpId(EntityMention mention) {
    String id = mention.getObjectId();
    int kbpPos = id.indexOf("KBP");
    if( kbpPos >= 0 )
      return Maybe.Just( id.substring(kbpPos + "KBP".length()) );
    else
      return Maybe.Nothing();
  }

  public static KBPEntity getKbpEntity(EntityMention mention) {
    return KBPNew.entName(mention.getFullValue())
                 .entType(getNERTag(mention).orCrash())
                 .entId(getKbpId(mention))
                 .KBPEntity();
  }

  static Pattern escaper = Pattern.compile("([^a-zA-z0-9])"); // should also include ,:; etc.?
  /**
   * Builds a new string where characters that have special meaning in Java regexes are escaped
   */
  public static String escapeSpecialRegexCharacters(String s) {
    return escaper.matcher(s).replaceAll("\\\\$1");
  }

  /** Format a memory value (in Bytes) using the appropriate units */
  public static String formatMemory(long mem) {
    String[] units = new String[]{ "B", "KB", "MB", "GB", "TB" };
    double asDouble = (double) mem;
    int unit = 0;
    while (asDouble > 1000.0) {
      unit += 1;
      asDouble /= 1000.0;
    }
    return new DecimalFormat("#.0").format(asDouble) + " " + units[unit];
  }

  /** Report memory usage, including used memory, free memory, total memory, and the maximum memory allotable to the JVM */
  public static String getMemoryUsage() {
    Runtime rt = Runtime.getRuntime();
    long max = rt.maxMemory();
    long total = rt.totalMemory();
    long free = rt.freeMemory();
    return String.format( "Memory used: %s, free: %s, total: %s, max: %s",
        formatMemory(total - free),
        formatMemory(free),
        formatMemory(total),
        formatMemory(max) );
  }

  /** Return the <b>most likely</b> (but not necessarily only) slot value type for a given relation */
  public static Maybe<NERTag> inferFillType(RelationType relation) {
    switch (relation) {
      case PER_ALTERNATE_NAMES:
        return  Maybe.Just(NERTag.PERSON);
      case PER_CHILDREN:
        return  Maybe.Just(NERTag.PERSON);
      case PER_CITIES_OF_RESIDENCE:
        return  Maybe.Just(NERTag.CITY);
      case PER_CITY_OF_BIRTH:
        return  Maybe.Just(NERTag.CITY);
      case PER_CITY_OF_DEATH:
        return  Maybe.Just(NERTag.CITY);
      case PER_COUNTRIES_OF_RESIDENCE:
        return  Maybe.Just(NERTag.COUNTRY);
      case PER_COUNTRY_OF_BIRTH:
        return  Maybe.Just(NERTag.COUNTRY);
      case PER_COUNTRY_OF_DEATH:
        return  Maybe.Just(NERTag.COUNTRY);
      case PER_EMPLOYEE_OF:
        return  Maybe.Just(NERTag.ORGANIZATION);
      case PER_MEMBER_OF:
        return  Maybe.Just(NERTag.ORGANIZATION);
      case PER_ORIGIN:
        return  Maybe.Just(NERTag.NATIONALITY);
      case PER_OTHER_FAMILY:
        return  Maybe.Just(NERTag.PERSON);
      case PER_PARENTS:
        return  Maybe.Just(NERTag.PERSON);
      case PER_SCHOOLS_ATTENDED:
        return  Maybe.Just(NERTag.ORGANIZATION);
      case PER_SIBLINGS:
        return  Maybe.Just(NERTag.PERSON);
      case PER_SPOUSE:
        return  Maybe.Just(NERTag.PERSON);
      case PER_STATE_OR_PROVINCES_OF_BIRTH:
        return  Maybe.Just(NERTag.STATE_OR_PROVINCE);
      case PER_STATE_OR_PROVINCES_OF_DEATH:
        return  Maybe.Just(NERTag.STATE_OR_PROVINCE);
      case PER_STATE_OR_PROVINCES_OF_RESIDENCE:
        return  Maybe.Just(NERTag.STATE_OR_PROVINCE);
      case PER_AGE:
        return  Maybe.Just(NERTag.NUMBER);
      case PER_DATE_OF_BIRTH:
        return  Maybe.Just(NERTag.DATE);
      case PER_DATE_OF_DEATH:
        return  Maybe.Just(NERTag.DATE);
      case PER_CAUSE_OF_DEATH:
        return  Maybe.Just(NERTag.CAUSE_OF_DEATH);
      case PER_CHARGES:
        return  Maybe.Just(NERTag.CRIMINAL_CHARGE);
      case PER_RELIGION:
        return  Maybe.Just(NERTag.RELIGION);
      case PER_TITLE:
        return  Maybe.Just(NERTag.TITLE);
      case ORG_ALTERNATE_NAMES:
        return  Maybe.Just(NERTag.ORGANIZATION);
      case ORG_CITY_OF_HEADQUARTERS:
        return  Maybe.Just(NERTag.CITY);
      case ORG_COUNTRY_OF_HEADQUARTERS:
        return  Maybe.Just(NERTag.COUNTRY);
      case ORG_FOUNDED_BY:
        return  Maybe.Just(NERTag.PERSON);
      case ORG_MEMBER_OF:
        return  Maybe.Just(NERTag.ORGANIZATION);
      case ORG_MEMBERS:
        return  Maybe.Just(NERTag.PERSON);
      case ORG_PARENTS:
        return  Maybe.Just(NERTag.ORGANIZATION);
      case ORG_POLITICAL_RELIGIOUS_AFFILIATION:
        return  Maybe.Just(NERTag.RELIGION);
      case ORG_SHAREHOLDERS:
        return  Maybe.Just(NERTag.PERSON);
      case ORG_STATE_OR_PROVINCES_OF_HEADQUARTERS:
        return  Maybe.Just(NERTag.STATE_OR_PROVINCE);
      case ORG_SUBSIDIARIES:
        return  Maybe.Just(NERTag.ORGANIZATION);
      case ORG_TOP_MEMBERS_SLASH_EMPLOYEES:
        return  Maybe.Just(NERTag.PERSON);
      case ORG_DISSOLVED:
        return  Maybe.Just(NERTag.DATE);
      case ORG_FOUNDED:
        return  Maybe.Just(NERTag.DATE);
      case ORG_NUMBER_OF_EMPLOYEES_SLASH_MEMBERS:
        return  Maybe.Just(NERTag.NUMBER);
      case ORG_WEBSITE:
        return Maybe.Just(NERTag.URL);
    }
    return Maybe.Nothing();
  }

  /**
   * Concatenate two arrays; also, a great example of Java Magic canceling out Java Failure.
   */
  public static <E> E[] concat(E[] a, E[] b) {
    @SuppressWarnings("unchecked") E[] rtn = (E[]) Array.newInstance(a.getClass().getComponentType(), a.length + b.length);
    System.arraycopy(a, 0, rtn, 0, a.length);
    System.arraycopy(b, 0, rtn, a.length, b.length);
    return rtn;
  }


  @SuppressWarnings({"ConstantConditions", "AssertWithSideEffects", "UnusedAssignment"})
  public static boolean assertionsEnabled() {
    boolean assertionsEnabled = false;
    assert assertionsEnabled = true;
    return assertionsEnabled;
  }


  public static boolean sameSlotFill(String candidate, String gold) {
    // Canonicalize strings
    candidate = candidate.trim().toLowerCase();
    gold = gold.trim().toLowerCase();
    // Special cases
    if (candidate == null || gold == null || candidate.trim().equals("")) {
      return false;
    }

    // Simple equality
    if (candidate.equals(gold)) {
      return true;
    }

    // Containment
    if (candidate.contains(gold) || gold.contains(candidate)) {
      return true;
    }

    // Else, give up
    return false;
  }


  /** I shamefully stole this from: http://rosettacode.org/wiki/Levenshtein_distance#Java --Gabor */
  public static int levenshteinDistance(String s1, String s2) {
    s1 = s1.toLowerCase();
    s2 = s2.toLowerCase();

    int[] costs = new int[s2.length() + 1];
    for (int i = 0; i <= s1.length(); i++) {
      int lastValue = i;
      for (int j = 0; j <= s2.length(); j++) {
        if (i == 0)
          costs[j] = j;
        else {
          if (j > 0) {
            int newValue = costs[j - 1];
            if (s1.charAt(i - 1) != s2.charAt(j - 1))
              newValue = Math.min(Math.min(newValue, lastValue), costs[j]) + 1;
            costs[j - 1] = lastValue;
            lastValue = newValue;
          }
        }
      }
      if (i > 0)
        costs[s2.length()] = lastValue;
    }
    return costs[s2.length()];
  }

  public static <E> int levenshteinDistance(E[] s1, E[] s2) {

    int[] costs = new int[s2.length + 1];
    for (int i = 0; i <= s1.length; i++) {
      int lastValue = i;
      for (int j = 0; j <= s2.length; j++) {
        if (i == 0)
          costs[j] = j;
        else {
          if (j > 0) {
            int newValue = costs[j - 1];
            if (!s1[i - 1].equals(s2[j - 1]))
              newValue = Math.min(Math.min(newValue, lastValue), costs[j]) + 1;
            costs[j - 1] = lastValue;
            lastValue = newValue;
          }
        }
      }
      if (i > 0)
        costs[s2.length] = lastValue;
    }
    return costs[s2.length];
  }

  public static Iterator<String> randomInsults() { return randomInsults(42); }

  /** http://iome.me/fz/need%20a%20random%20sentence?/why%20not%20zoidberg? */
  public static Iterator<String> randomInsults(final int seed) {
    final HashSet<String> seen = new HashSet<>();
    final String[] adj1 = {"warped", "babbling", "madcap", "whining", "wretched", "loggerheaded", "threadbare", "foul", "artless",
        "artless", "baudy", "beslumbering", "bootless", "churlish", "clouted", "craven", "dankish", "dissembling", "droning",
        "errant", "fawning", "fobbing", "froward", "frothy", "gleeking", "goatish", "gorbellied", "impertinent", "infectious",
        "jarring", "loggerheaded", "lumpish", "mammering", "mangled", "mewling", "paunchy", "pribbling", "puking", "puny",
        "qualling", "rank", "reeky", "roguish", "rutting", "saucy", "spleeny", "spongy", "surly", "tottering", "unmuzzled", "vain",
        "venomed", "villainous", "warped", "wayward", "weedy", "yeasty" };
    final String[] adj2 = {"toad-spotted", "guts-griping", "beef-witted", "ill-favored", "hare-brained", "fat-kidneyed", "white-bearded",
        "shrill-voiced", "base-court", "bat-fowling", "beef-witted", "beetle-headed", "boil-brained", "clapper-clawed",
        "clay-brained", "common-kissing", "crook-pated", "dismal-dreaming", "dizzy-eyed", "doghearted", "dread-bolted", "earth-vexing",
        "elf-skinned", "fat-kidneyed", "fen-sucked", "flap-mouthed", "fly-bitten", "folly-fallen", "fool-born", "full-gorged",
        "guts-griping", "half-faced", "hasty-witted", "hedge-born", "hell-hated", "idle-headed", "ill-breeding", "ill-nurtured",
        "knotty-pated", "milk-livered", "motley-minded", "onion-eyed", "plume-plucked", "pottle-deep", "pox-marked", "reeling-ripe",
        "rough-hewn", "rude-growing", "rump-fed", "shard-borne", "sheep-biting", "spur-galled", "swag-bellied", "tardy-gaited",
        "tickle-brained", "toad-spotted", "unchin-snouted", "weather-bitten" };
    final String[] nouns = {"jolt-head", "mountain-goat", "fat-belly", "malt-worm", "minnow", "so-and-so", "maggot-pie", "foot-licker", "land-fish",
        "apple-john", "baggage", "barnacle", "bladder", "boar-pig", "bugbear", "bum-bailey", "canker-blossom", "clack-dish", "clotpole",
        "coxcomb", "codpiece", "death-token", "dewberry", "flap-dragon", "flax-wench", "flirt-gill", "foot-licker", "fustilarian",
        "giglet", "gudgeon", "haggard", "harpy", "hedge-pig", "horn-beast", "hugger-mugger", "joithead", "lewdster", "lout", "maggot-pie",
        "malt-worm", "mammet", "measle", "minnow", "miscreant", "moldwarp", "mumble-news", "nut-hook", "pigeon-egg", "pignut",
        "puttock", "pumpion", "ratsbane", "scut", "skainsmate", "strumpet", "varlot", "vassal", "whey-face", "wagtail", };
    return new Iterator<String> () {
      private Random random = new Random(seed);
      @Override
      public boolean hasNext() {
        return true;
      }
      @Override
      public String next() {
        String a1 = adj1[random.nextInt(adj1.length)];
        String a2 = adj2[random.nextInt(adj2.length)];
        String n  = nouns[random.nextInt(nouns.length)];
        String candidate = "Curse thee, KBP, thou " + a1 + " " + a2 + " " + n;
        if (seen.contains(candidate)) { return next(); } else { seen.add(candidate); return candidate; }
      }
      @Override
      public void remove() {
      }
    };
  }


  public static Maybe<Double> noisyOr(Maybe<Double> score1, Maybe<Double> score2) {
    if( score1.isNothing() ) return score2;
    else if( score2.isNothing() ) return score1;
    else return Maybe.Just( 1 - (1 - score1.get()) * (1-score2.get()) );
  }



  private static final AtomicInteger entityMentionCount = new AtomicInteger(0);
  public static String makeEntityMentionId(Maybe<String> kbpId) {
    String id = "EM" + entityMentionCount.incrementAndGet();
    for (String idImpl : kbpId) { id += "-KBP" + idImpl; }
    return id;
  }

  /** Determine if a slot is close enough to any entity to be considered a valid candidate */
  public static boolean closeEnough(Span slotSpan, Collection<Span> entitySpans) {
    if (entitySpans.isEmpty()) { return true; }
    for (Span entitySpan : entitySpans) {
      if (slotSpan.end() <= entitySpan.start()
          && entitySpan.start() - slotSpan.end() < Props.MAX_DISTANCE_BETWEEN_ENTITY_AND_SLOT) {
        return true;
      } else if (entitySpan.end() <= slotSpan.start()
          && slotSpan.start() - entitySpan.end() < Props.MAX_DISTANCE_BETWEEN_ENTITY_AND_SLOT) {
        return true;
      }
    }
    return false;
  }

  public static String noSpecialChars(String original) {
    char[] chars = original.toCharArray();
    // Compute the size of the output
    int size = 0;
    boolean isAllLowerCase = true;
    for (char aChar : chars) {
      if (aChar != '\\' && aChar != '"' && aChar != '-') {
        if (isAllLowerCase && !Character.isLowerCase(aChar)) { isAllLowerCase = false; }
        size += 1;
      }
    }
    if (size == chars.length && isAllLowerCase) { return original; }
    // Copy to a new String
    char[] out = new char[size];
    int i = 0;
    for (char aChar : chars) {
      if (aChar != '\\' && aChar != '"' && aChar != '-') {
        out[i] = Character.toLowerCase(aChar);
        i += 1;
      }
    }
    // Return
    return new String(out);
  }
  
  public static boolean nearExactEntityMatch( String higherGloss, String lowerGloss ) {
    // case: slots have same relation, and that relation isn't an alternate name
    // Filter case sensitive match
    if (higherGloss.equalsIgnoreCase(lowerGloss)) { return true; }
    // Ignore certain characters
    else if (noSpecialChars(higherGloss).equalsIgnoreCase(noSpecialChars(lowerGloss))) { return true; }
    return false;
  }


  public static boolean approximateEntityMatch( KBPEntity entity, KBPEntity otherEntity ) {
    return Props.ENTITYLINKING_LINKER.get().sameEntity( new EntityContext(entity), new EntityContext(otherEntity) );
  }

  private static String removeDisallowedAlternateNameVariants(String in) {
    return in.toLowerCase().replaceAll("corp.?", "").replaceAll("llc.?", "").replaceAll("inc.?", "")
        .replaceAll("\\s+", " ").trim();
  }

  public static boolean isValidAlternateName(String name1, String name2) {
    return !removeDisallowedAlternateNameVariants(name1).equals(removeDisallowedAlternateNameVariants(name2));

  }

  public static Maybe<Integer> getNumericValue( KBPSlotFill candidate ) {
    // Case: Rewrite mentions to their antecedents (person and organization)
    // If we already have provenance...
    if (candidate.provenance.isDefined() && candidate.provenance.get().containingSentenceLossy.isDefined() &&
            candidate.provenance.get().slotValueMentionInSentence.isDefined() ) {
      CoreMap lossySentence = candidate.provenance.get().containingSentenceLossy.get();
      Span slotSpan = candidate.provenance.get().slotValueMentionInSentence.get();
      List<CoreLabel> tokens = lossySentence.get(CoreAnnotations.TokensAnnotation.class);
      Maybe<List<CoreLabel>> provenanceSpan = Maybe.Just(tokens.subList(slotSpan.start(), slotSpan.end()));

      for (List<CoreLabel> valueSpan : provenanceSpan) {
        for (CoreLabel token : valueSpan) {
          if (token.containsKey(CoreAnnotations.NumericValueAnnotation.class)) {
            return Maybe.Just(token.get(CoreAnnotations.NumericValueAnnotation.class).intValue());
          }
        }
      }
    }
    // I'm trying my best here
    return Maybe.Just(NumberNormalizer.wordToNumber(candidate.key.slotValue).intValue());
    // TODO(arun) Catch the case where this isn't a number
  }

  /**
   * Tries very hard to match a given sentence fragment to its original sentence.
   * Whitespace is ignored on both the sentence tokenization and the gloss.
   * @param sentence The sentence to match against. The returned span will be in token offsets into this sentence
   * @param gloss The string to fit into the sentence somewhere.
   * @param guess An optional span that denotes where we think this gloss should go -- the closest matching gloss
   *              to this is returned.
   * @return Our best guess of where the gloss came from in the original sentence, or {@link Maybe#Nothing} if we couldn't find anything.
   */
  public static Maybe<Span> getTokenSpan(char[][] sentence, char[] gloss, Maybe<Span> guess) {
    // State (think finite state machine with multiple "heads" tracking progress)
    boolean[] heads = new boolean[gloss.length];
    int[] starts = new int[gloss.length];
    // Matches
    List<Span> finishedSpans = new ArrayList<>();
    // Initialize State
    Arrays.fill(heads, false);

    // Run FSA Matcher
    for (int tokI = 0; tokI < sentence.length; ++tokI) {
      for (int charI = 0; charI < sentence[tokI].length; ++charI) {
        // Initialize
        starts[0] = tokI;
        heads[0] = true;

        // (1) Scroll through whitespace
        for (int onPriceI = 1; onPriceI < gloss.length; ++onPriceI) {
          if (heads[onPriceI - 1] && Character.isWhitespace(gloss[onPriceI - 1])) {
            heads[onPriceI] = true;
            starts[onPriceI] = starts[onPriceI - 1];
          }
        }
        // (2) Check if we've whitespace'd to the end
        if (heads[gloss.length - 1] && Character.isWhitespace(gloss[gloss.length - 1])) {
          assert starts[gloss.length - 1] >= 0;
          finishedSpans.add(new Span(starts[gloss.length - 1], charI == 0 ? tokI : tokI + 1));
        }

        // (3) try for an exact match
        for (int onPriceI = gloss.length - 1; onPriceI >= 0; --onPriceI) {
          if (heads[onPriceI] || onPriceI == 0) {
            // Case: found an active partial match to potentially extend
            if (sentence[tokI][charI] == gloss[onPriceI]) {
              // Case: literal match
              if (onPriceI >= gloss.length - 1) {
                // Finished match
                assert starts[onPriceI] >= 0;
                finishedSpans.add(new Span(starts[onPriceI], tokI + 1));
              } else {
                // Move head
                heads[onPriceI + 1] = true;
                starts[onPriceI + 1] = starts[onPriceI];
              }
              if (!Character.isWhitespace(sentence[tokI][charI])) {
                // Either we matched whitespace, or invalidate this position
                heads[onPriceI] = false;
                starts[onPriceI] = -1;
              }
            }
          }
        }

        // (4) Scroll through whitespace (and potentially kill it!)
        for (int onPriceI = 1; onPriceI < gloss.length; ++onPriceI) {
          if (heads[onPriceI - 1] && Character.isWhitespace(gloss[onPriceI - 1])) {
            heads[onPriceI] = true;
            starts[onPriceI] = starts[onPriceI - 1];
          } else if (!heads[onPriceI - 1] && Character.isWhitespace(gloss[onPriceI - 1]) &&
              !Character.isWhitespace(sentence[tokI][charI])) {
            heads[onPriceI] = false;
            starts[onPriceI] = -1;
          }
        }
        // (5) Check if we've whitespace'd to the end
        if (heads[gloss.length - 1] && Character.isWhitespace(gloss[gloss.length - 1])) {
          assert starts[gloss.length - 1] >= 0;
          finishedSpans.add(new Span(starts[gloss.length - 1], tokI + 1));
        }
      }
    }

    // Find closest match (or else first match)
    // Shortcuts
    if (finishedSpans.size() == 0) { return Maybe.Nothing(); }
    if (finishedSpans.size() == 1) { return Maybe.Just(finishedSpans.get(0)); }
    if (guess.isDefined()) {
      // Case; find closest span
      Span toReturn = null;
      int min = Integer.MAX_VALUE;
      for (Span candidate : finishedSpans) {
        int distance = Math.abs(candidate.start() - guess.get().start()) + Math.abs(candidate.end() - guess.get().end());
        if (distance < min) {
          min = distance;
          toReturn = candidate;
        }
      }
      return Maybe.Just(toReturn);
    } else {
      // Case:
      return Maybe.Just(finishedSpans.get(0));
    }
  }

  /**
   * @see Utils#getTokenSpan(char[][], char[], Maybe)
   */
  private static Maybe<Span> getTokenSpan(List<CoreLabel> sentence, String glossString, Maybe<Span> guess, Class<? extends CoreAnnotation<String>> textAnnotation) {
    // Get characters from sentence
    char[][] tokens = new char[sentence.size()][];
    for (int i = 0; i < sentence.size(); ++i) {
      @SuppressWarnings("unchecked") char[] tokenChars = ((String) sentence.get(i).get((Class) textAnnotation)).toCharArray();
      tokens[i] = tokenChars;
    }
    // Call low-level [tested] implementation
    return getTokenSpan(tokens, glossString.toCharArray(), guess);

  }

  /**
   * Gets the best matching token span in the given sentence for the given gloss, with an optional
   * guessed span to help the algorithm out.
   *
   * @see Utils#getTokenSpan(char[][], char[], Maybe)
   */
  public static Maybe<Span> getTokenSpan(List<CoreLabel> sentence, String gloss, Maybe<Span> guess) {
    Maybe<Span> span = getTokenSpan(sentence, gloss, guess, CoreAnnotations.OriginalTextAnnotation.class);
    if (!span.isDefined()) {
      span = getTokenSpan(sentence, gloss, guess, CoreAnnotations.TextAnnotation.class);
    }
    if (span.isDefined()) {
      assert span.get().start() < sentence.size();
      assert span.get().end() <= sentence.size();
    }
    return span;
  }


  /**
   * A utility data structure.
   * @see Utils#sortRelationsByPrior(java.util.Collection)
   */
  private static final Counter<String> relationPriors = new ClassicCounter<String>(){{
    for (RelationType rel : RelationType.values()) {
      setCount(rel.canonicalName, rel.priorProbability);
    }
  }};

  /**
   * Returns a sorted list of relations names, ordered by their prior probability.
   * This is guaranteed to return a stable order.
   * @param relations A collection of relations to sort.
   * @return A sorted list of the relations in descending order of prior probability.
   */
  public static List<String> sortRelationsByPrior(Collection<String> relations) {
    List<String> sorted = new ArrayList<>(relations);
    Collections.sort(sorted, (o1, o2) -> {
      double count1 = relationPriors.getCount(o1);
      double count2 = relationPriors.getCount(o2);
      if (count1 < count2) { return 1; }
      if (count2 < count1) { return -1; }
      return o1.compareTo(o2);
    });
    return sorted;
  }

  /**
   * Checks if the given path <b>is</b> a loop. Note that there can be subloops within the path -- this is not checked for
   * by this function (see {@link Utils#doesLoopPath(java.util.Collection)}).
   * @param path The path, as a list of {@link KBPSlotFill}s, each of which represents an edge.
   * @return True if this path represents a complete loop.
   */
  @Deprecated // TODO(gabor) this is broken!
  public static boolean isLoopPath(List<KBPSlotFill> path) {
    return isLoop(CollectionUtils.lazyMap(path, in -> in.key));
  }

  /**
   * Checks if the given path <b>is</b> a loop. Note that there can be subloops within the path -- this is not checked for
   * by this function (see {@link Utils#doesLoopPath(java.util.Collection)}).
   * @param path The path, as a list of {@link KBTriple}s, each of which represents an edge.
   * @return True if this path represents a complete loop.
   */
  @Deprecated // TODO(gabor) this is broken!
  public static boolean isLoop(List<KBTriple> path) {
    if (!doesLoop(path)) { return false; }
    if (path.size() < 2) { return false; }
    // Get the first entity in the path
    KBPEntity firstEntity = path.get(0).getEntity();
    if (path.get(1).getEntity().equals(firstEntity) && !path.get(1).getSlotEntity().equalsOrElse(path.get(0).getSlotEntity().orNull(), false)) {
      firstEntity = path.get(0).getSlotEntity().orNull();
    }
    if (firstEntity == null) { return false; }
    // Get the last entity in the path
    KBPEntity lastEntity = path.get(path.size() - 1).getSlotEntity().orNull();
    if (lastEntity == null || path.get(path.size() - 2).getSlotEntity().equalsOrElse(lastEntity, false)) {
      lastEntity = path.get(path.size() - 1).getEntity();
    }
    // See if they match
    return firstEntity.equals(lastEntity);

  }

  /**
   * Checks if the given path <b>contains</b> a loop.
   * To check if a path is itself a loop, use {@link Utils#isLoopPath(java.util.List)}}.
   * @param path The path, as a set of {@link KBPSlotFill}s, each of which represents an edge.
   * @return True if this path contains a loop
   */
  public static boolean doesLoopPath(Collection<KBPSlotFill> path) {
    return doesLoop(CollectionUtils.lazyMap(path, in -> in.key));
  }

  /**
   * Checks if the given path <b>contains</b> a loop.
   * To check if a path is itself a loop, use {@link Utils#isLoop(java.util.List)}}.
   * @param path The path, as a set of {@link KBTriple}s, each of which represents an edge.
   * @return True if this path contains a loop
   */
  public static boolean doesLoop(Collection<KBTriple> path) {
    Set<KBPEntity> entitiesInPath = new HashSet<>();
    for (KBTriple fill : path) {
      entitiesInPath.add(fill.getEntity());
      entitiesInPath.add(fill.getSlotEntity().orNull());
    }
    return entitiesInPath.size() <= path.size();
  }

  /**
   * Return all clauses which are valid antecedents for the passed clause.
   * For example, on input path A->B->C->A, it would return (A->B)+(B->C) and (A->B)+(C->A) and (B->C)+(C->A).
   * @param clause The input clause to find antecedents for.
   * @param normalize If true, normalize the resulting clauses, abstracting out entities (or renormalizing
   *                  to canonical form if the entities are already abstracted)
   * @return A set of valid antecedents, which should be added to the set of paths seen.
   */
  private static Set<Set<KBTriple>> getValidAntecedents(Collection<KBTriple> clause, boolean normalize) {
    assert Utils.doesLoop(clause);
    Set<Set<KBTriple>> rtn = new HashSet<>();
    HashSet<KBTriple> mutableClause = new HashSet<>(clause);
    for (KBTriple consequent : clause) {  // for each candidate consequent
      mutableClause.remove(consequent);
      // This block just checks if both entities in the consequent are present somewhere
      // in the antecedents
      KBPEntity e1 = consequent.getEntity();
      KBPEntity e2 = consequent.getSlotEntity().orNull();  // careful with equals() order here
      boolean hasE1 = false;
      boolean hasE2 = false;
      for (KBTriple antecedent : mutableClause) {
        KBPEntity a1 = antecedent.getEntity();
        Maybe<KBPEntity> a2 = antecedent.getSlotEntity();
        if (a1.equals(e1) || a2.equalsOrElse(e1, false)) { hasE1 = true; }
        if (a1.equals(e2) || a2.equalsOrElse(e2, false)) { hasE2 = true; }
      }
      // If this is true, add the antecedents for this consequent as a valid antecedent
      if (hasE1 && hasE2) {
        if (!Utils.doesLoop(mutableClause)) {
          assert !Utils.doesLoop(mutableClause);
          rtn.add(normalize ? normalizeConjunction(mutableClause) : new HashSet<>(mutableClause));
        }
      }
      mutableClause.add(consequent);
    }
    return rtn;
  }

  /** @see Utils#getValidAntecedents(java.util.Collection, boolean) */
  public static Set<Set<KBTriple>> getValidAntecedents(Collection<KBTriple> clause) {
    return getValidAntecedents(clause, false);
  }

  /** @see Utils#getValidAntecedents(java.util.Collection, boolean) */
  public static Set<Set<KBTriple>> getValidNormalizedAntecedents(Collection<KBTriple> clause) {
    return getValidAntecedents(clause, true);
  }

  /**
   * Replace the variables of an inferential path with canonical variable names (x0, x1, ...)
   * @param conjunction - Conjunction of paths
   * @return - conjunction of facts with variables replaced.
   */
  public static Pair<Double, ? extends Set<KBTriple>> abstractConjunction(Collection<KBPSlotFill> conjunction) {
    Collection<KBTriple> triples = new ArrayList<>();
    double score = 0.0;
    for (KBPSlotFill fill : conjunction) {
      triples.add(fill.key);
      score = score * fill.score.getOrElse(1.0);
    }
    return Pair.makePair(score, normalizeConjunction(triples));
  }

  /** @see Utils#normalizeConjunction(java.util.Collection, java.util.Map)  */
  public static LinkedHashSet<KBTriple> normalizeConjunction(Collection<KBTriple> conjunction) {
    return normalizeConjunction(conjunction, new HashMap<KBPEntity, String>());
  }

  /**
   * <p>Takes a potentially already abstracted conjunction, and normalizes it.
   * This is particularly relevant if the conjunction is a subset of a larger conjunction,
   * and the variable names may not agree (e.g., it may start on x_1 rather than x_0).</p>
   *
   * <p>This functionality is a pain in the butt. Please don't judge me...</p>
   *
   * @param conjunction Conjunction of paths to normalize, potentially already normalized once.
   *                    This input will not be mutated.
   * @param mapping An optional return variable to store the entity to variable name mapping
   * @return conjunction of facts with variables replaced. This is guaranteed to always be in a
   *           canonical form. This set is a copy of the input conjunction.
   */
  public static LinkedHashSet<KBTriple> normalizeConjunction(Collection<KBTriple> conjunction, Map<KBPEntity, String> mapping) {
    Set<KBTriple> abstractConjunction = new HashSet<>();

    // Case: there are multiple fills with the same relation; disambiguate.
    Map<String, Pair<List<KBTriple>, Map<KBPEntity, String>>> candidates = new HashMap<>();
    // Collect candidates
    for (List<KBTriple> permutation : CollectionUtils.permutations(conjunction)) {
      List<KBTriple> normalized = new ArrayList<>();
      Map<KBPEntity, String> mapper = new HashMap<>();
      for (KBTriple fill : permutation) {
        normalized.add(tryNormalize(fill, true, true, mapper));
      }
      candidates.put(StringUtils.join(normalized, "^"), Pair.makePair(normalized, mapper));
    }
    // Sort candidates
    List<String> keys = new ArrayList<>(candidates.keySet());
    Collections.sort(keys);
    String firstKey = keys.get(0);

    // Return
    mapping.putAll(candidates.get(firstKey).second);
    return new LinkedHashSet<>(candidates.get(firstKey).first);
  }

  /**
   * Normalize a conjunction, keeping the variable ordering in the input.
   *
   * @param conjunction The conjunction, in a relevant order.
   * @return The normalized conjunction, in the order passed as input.
   */
  public static List<KBTriple> normalizeOrderedConjunction(List<KBTriple> conjunction) {
    final Map<KBPEntity, String> variables = new HashMap<>();
    return CollectionUtils.map(conjunction, new Function<KBTriple, KBTriple>() {
      private String var(KBPEntity entity) {
        if (!variables.containsKey(entity)) {
          variables.put(entity, "x" + variables.size());
        }
        return variables.get(entity);
      }

      @Override
      public KBTriple apply(KBTriple in) {
        return KBPNew.from(in).entName(var(in.getEntity())).slotValue(var(in.getSlotEntity().orCrash())).KBTriple();
      }
    });
  }

  /**
   * <p>
   *   Normalize a conjunction for use as a directed entailment, where the last element of the list
   *   is the consequent and the first elements are the antecedents.
   * </p>
   *
   * <p>
   *   This is a special case, as we'd like equivalent antecedents to be abstracted to identical formulas,
   *   however would like the consequent to always remain the last element in the list, and always conform
   *   to the abstraction of the antecedents.
   * </p>
   *
   * @param conjunction The entailment conjunction; the first n-1 elements are the antecedents, the last element is the consequent.
   * @return An ordered abstracted tuple corresponding to the conjunction in this entailment.
   */
  public static List<KBTriple> normalizeEntailment(List<KBTriple> conjunction) {
    // Deterministically ground antecedent
    Map<KBPEntity, String> variables = new HashMap<>();
    LinkedHashSet<KBTriple> antecedent = normalizeConjunction(conjunction.subList(0, conjunction.size() - 1), variables);

    // Create the full clause
    KBTriple consequentUnnormalized = conjunction.get(conjunction.size() - 1);
    if (!variables.containsKey(consequentUnnormalized.getEntity())) {
      variables.put(consequentUnnormalized.getEntity(), "x" + variables.size());
    }
    if (!variables.containsKey(consequentUnnormalized.getSlotEntity().orCrash())) {
      variables.put(consequentUnnormalized.getSlotEntity().orCrash(), "x" + variables.size());
    }
    KBTriple consequent =  KBPNew.from(consequentUnnormalized)
        .entName(variables.get(consequentUnnormalized.getEntity()))
        .slotValue(variables.get(consequentUnnormalized.getSlotEntity().orCrash())).KBTriple();

    // Create return
    ArrayList<KBTriple> rtn = new ArrayList<>(antecedent);
    rtn.add(consequent);
    assert rtn.get(rtn.size() - 1).relationName.equals(conjunction.get(conjunction.size() - 1).relationName);
    return rtn;
  }

  /**
   * A helper for {@link Utils#normalizeConjunction(java.util.Collection)}.
   */
  private static KBTriple tryNormalize(KBTriple fill, boolean canAddEntity, boolean canAddSlotValue, Map<KBPEntity, String> mapper) {
    KBPEntity head = fill.getEntity();
    KBPEntity tail = fill.getSlotEntity().orCrash();

    if (!canAddEntity && !mapper.containsKey(head)) { return null; }
    if (!canAddSlotValue && !mapper.containsKey(tail)) { return null; }

    if(! mapper.containsKey(head)) mapper.put( head, "x" + mapper.size() );
    if(! mapper.containsKey(tail)) mapper.put( tail, "x" + mapper.size() );

    return KBPNew
        .entName(mapper.get(head))
        .entType(head.type)
        .slotValue(mapper.get(tail))
        .slotType(tail.type)
        .rel(fill.relationName).KBTriple();
  }

  /**
   * Determine if the given String is an integer or not.
   */
  public static boolean isInteger(String string) {
    try {
      Integer.parseInt(string);
      return true;
    } catch (NumberFormatException e) {
      return false;
    }
  }

  /**
   * Create a subject field for an email, summarizing the state and results of this KBP run
   *
   * @param score The (possible) results reported for this run.
   * @param error A possible exception that was encountered during the run.
   * @return A plain-text subject.
   */
  public static String mkEmailSubject(Maybe<KBPScore> score, Maybe<? extends Throwable> error) {
    StringBuilder subject = new StringBuilder();
    // Register errors
    for (Throwable t : error) {
      subject.append("{FAILED} ");
    }
    // Register run directory
    subject.append(Props.KBP_YEAR.name()).append("@").append(Props.WORK_DIR.getName());
    // Register performance
    DecimalFormat df = new DecimalFormat("00.0");
    for (KBPScore s : score) {
      Triple<Double, Double, Double> optimal = s.optimalPrecisionRecallF1();
      subject.append(" (P:").append(df.format(optimal.first * 100))
          .append(" R:").append(df.format(optimal.second * 100))
          .append(" F1:").append(df.format(optimal.third * 100))
          .append(")");
    }
    subject.append(": ").append(Props.KBP_DESCRIPTION);
    // Return
    return subject.toString();
  }

  /**
   * Create an email message summarizing this run. This includes both the parameters for the run,
   * as well as the (possible) results reported.
   *
   * @param score The (possible) results reported for this run.
   * @param error A possible exception that was encountered during the run.
   * @return An HTML email body encoding the parameters and results of this run.
   */
  public static String mkEmailBody(Maybe<KBPScore> score, Maybe<? extends Throwable> error, Date startTime) {
    Date endTime = Calendar.getInstance().getTime();
    StringBuilder body = new StringBuilder();

    // Compute changed results
    Map<String, String> options = Props.asMap();
    Map<String, String> changedOptions = new HashMap<>();
    Map<String, String> unchangedOptions = new HashMap<>();
    for (Map.Entry<String, String> entry : options.entrySet()) {
      if (entry.getValue() != null) {
        if (entry.getKey().startsWith("cache.") ||  // don't care about caching options
            entry.getKey().startsWith("process.domreader.") ||  // don't care about domreader options
            (Props.defaultOptions.get(entry.getKey()) != null && Props.defaultOptions.get(entry.getKey()).startsWith("edu/stanford/nlp")) ||  // overwriting classpaths also isn't very interesting
            (Props.defaultOptions.get(entry.getKey()) != null && Props.defaultOptions.get(entry.getKey()).equals("/dev/null")) ||  // also not very interesting
            entry.getValue().equals(Props.defaultOptions.get(entry.getKey()))) {
          unchangedOptions.put(entry.getKey(), entry.getValue());
        } else {
          changedOptions.put(entry.getKey(), entry.getValue());
        }
      }
    }

    // Write header
    body.append("<center><h2>KBP Results</h2></center>\n")
        .append("<p><b>Runtime:</b> ").append(startTime.toString()).append(" to ").append(endTime.toString()).append("</p>\n\n");

    // Write changed options
    if (!changedOptions.isEmpty()) {
      body.append("<h3>Likely Interesting Options</h3>\n");
      body.append("<ul style=\"list-style-type:none;\">");
      for (Map.Entry<String, String> option : changedOptions.entrySet()) {
        body.append("<li><b>").append(option.getKey()).append("</b> &nbsp;&nbsp; <i>").append(option.getValue().substring(0, Math.min(80 - option.getKey().length(), option.getValue().length()))).append("</i>\n");
      }
      body.append("</ul>");
    }

    // Write errors
    for (Throwable t : error) {
      body.append("<h3>Errors</h3>\n");
      body.append("<font color=\"red\">\n");
      Throwable onPrice = t;
      while (onPrice != null) {
        body.append("<b>").append(onPrice.getClass().getName()).append("</b>&nbsp;&nbsp;<i>").append(Maybe.fromNull(onPrice.getMessage()).getOrElse("")).append("</i>\n");
        body.append("<ul style=\"list-style-type:none;\">");
        StackTraceElement[] stackTrace = onPrice.getStackTrace();
        for (StackTraceElement elem : stackTrace) {
          body.append("<li>").append(elem.toString()).append("</li>\n");
        }
        body.append("</ul>\n");
        onPrice = onPrice.getCause();
      }
      body.append("</font>\n");
    }

    // Write score
    for (KBPScore s : score) {
      DecimalFormat percent = new DecimalFormat("00.000%");
      Triple<Double, Double, Double> optimal = s.optimalPrecisionRecallF1();
      body.append("<h3>Score</h3>\n");
      body.append("<table style=\"font-family: verdana,arial,sans-serif; font-size:12px; color:#333333;\">\n");
      body.append("<tr style=\"border-bottom: 1px solid #ffffff\"><th>Metric</th><th>Value</th></tr>\n");
      body.append("<tr><td>Precision:</td><td>").append(percent.format(s.precision)).append("</td></tr>\n");
      body.append("<tr><td>Recall:</td><td>").append(percent.format(s.recall)).append("</td></tr>\n");
      body.append("<tr><td><b>F1:</b></td><td><b>").append(percent.format(s.f1)).append("</b></td></tr>\n");
      body.append("<tr style=\"border-bottom: 1px solid #C0C0C0; border-top: 1px solid #C0C0C0;\"><td></td><td></td></tr>\n");
      body.append("<tr><td>Optimal Precision:</td><td>").append(percent.format(optimal.first)).append("</td></tr>\n");
      body.append("<tr><td>Optimal Recall:</td><td>").append(percent.format(optimal.second)).append("</td></tr>\n");
      body.append("<tr><td><b>Optimal F1:</b></td><td><b>").append(percent.format(optimal.third)).append("</b></td></tr>\n");
      body.append("<tr style=\"border-bottom: 1px solid #C0C0C0; border-top: 1px solid #C0C0C0;\"><td></td><td></td></tr>\n");
      body.append("<tr><td>Area Under PR Curve:</td><td>").append(new DecimalFormat("0.000").format(s.areaUnderPRCurve())).append("</td></tr>\n");
      body.append("</table>\n");
    }

    // Write unchanged options
    body.append("<h3>Other Options</h3>\n");
    body.append("<ul style=\"list-style-type:none;\">");
    for (Map.Entry<String, String> option : unchangedOptions.entrySet()) {
      body.append("<li>").append(option.getKey()).append(" &nbsp;&nbsp; <i>").append(option.getValue().substring(0, Math.min(80 - option.getKey().length(), option.getValue().length()))).append("</i>\n");
    }
    body.append("</ul>");

    return body.toString()
        .replaceAll("<th>", "<th style=\"padding: 1px; background-color: #dedede;\">")
        .replaceAll("<tr><td>", "<tr><td style=\"padding: 1px; background-color: #ffffff; text-align: right;\">")
        .replaceAll("</td><td>", "</td><td style=\"padding: 1px; background-color: #ffffff; text-align: left;\">");
  }

  /**
   * A really dumb utility to print out the difference between two entity graphs.
   * Useful for, e.g., comparing the output of two inference methods.
   * @param expected The "expected" output. Entries not in here will be in red.
   * @param actual The "actual" output. Entries not in here will be in yellow.
   * @param queryEntity The query entity to collect slot fills from.
   */
  public static void printDiff(EntityGraph expected, EntityGraph actual, KBPEntity queryEntity) {
    startTrack("Output Diff (" + queryEntity + ")");
    Set<KBPSlotFill> expectedFills = new HashSet<>(expected.getOutgoingEdges(queryEntity));
    Set<KBPSlotFill> actualFills = new HashSet<>(actual.getOutgoingEdges(queryEntity));
    for (KBPSlotFill fill : actualFills) {
      if (!fill.key.hasKBPRelation()) { continue; }
      if (!expectedFills.contains(fill)) {
        Redwood.log(Color.RED.apply(fill.toString()));
      }
    }
    for (KBPSlotFill fill : expectedFills) {
      if (!fill.key.hasKBPRelation()) { continue; }
      if (!actualFills.contains(fill)) {
        Redwood.log(Color.YELLOW.apply(fill.toString()));
      }
    }
    endTrack("Output Diff (" + queryEntity + ")");
  }
}
