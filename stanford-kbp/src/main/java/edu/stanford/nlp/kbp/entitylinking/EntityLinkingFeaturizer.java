package edu.stanford.nlp.kbp.entitylinking;

import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.trees.HeadFinder;
import edu.stanford.nlp.trees.ModCollinsHeadFinder;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.util.ArrayUtils;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IntPair;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.logging.Redwood;
import org.apache.commons.lang.StringUtils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Serializable;
import java.util.*;

/**
 * A featurizer for pairwise entity linking.
 *
 * @author Melvin Jose
 */
public class EntityLinkingFeaturizer implements Serializable{

  private class Option<T> {
    private T obj;
    public Option(T obj){ this.obj = obj; }
    public Option(){};
    public T get(){ return obj; }
    public void set(T obj){ this.obj = obj; }
    public boolean exists(){ return obj != null; }
  }

  //redwood
  private static final Redwood.RedwoodChannels logger = Redwood.channels("ELF");

  private static long serialVersionUID =1l;

  /** A map from male names to their canonical form; e.g., Ron -&gt; Aaron */
  protected final Map<String, String> maleNamesLowerCase;
  /** A map from female names to their canonical form; e.g., Abby -&gt; Abigail */
  protected final Map<String, String> femaleNamesLowerCase;
  /** A map from word to its abbreviation; e.g., California -&gt; Ca. */
  protected final Map<String, String> abbreviations;
  /** An array of feature counts in the data */
  protected final int[] countsFeature;

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
    add("syndicate"); add("institute"); add("fund"); add("foundation"); add("club"); add("partners"); add("group");
  }});

  private static final HeadFinder headFinder = new ModCollinsHeadFinder();

  /**
   * A short set of determiners.
   */
  public static final Set<String> DETERMINERS = Collections.unmodifiableSet(new HashSet<String>() {{
    add("the"); add("The"); add("a"); add("A");
  }});

  public static final Set<String> PUNCTUATIONS = Collections.unmodifiableSet(new HashSet<String>() {{
    add(","); add(":"); add(";"); add("."); add("'");
  }});
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
  protected String[] stripDeterminers(String[] tokens) {

    for (int i=0; i < tokens.length; i++) {
      if(DETERMINERS.contains(tokens[i])) {
        ArrayUtils.removeAt(tokens, i);
      }
    }
    return tokens;
  }

  /**
   *
   * @param input
   * @return a string with punctuations removed
   */
  protected String stripPunctuations(String input) {
    for (String determiner : PUNCTUATIONS) {
      if (input.startsWith(determiner)) { input = input.substring(determiner.length()).trim(); }
    }
    return input.trim();
  }

  /**
   *
   * @param tokens
   * @return tokens with punctuations removed
   */
  protected String[] stripPunctuations(String[] tokens) {

    for (int i=0; i < tokens.length; i++) {
      if(PUNCTUATIONS.contains(tokens[i])) {
        ArrayUtils.removeAt(tokens, i);
      }
    }
    return tokens;
  }


  protected boolean isAcronym(String[] oneToks, String[] twoToks) {
    if(AcronymMatcher.isAcronym(oneToks, twoToks)) {
      //countsFeature[8]++;
      return true;
    } else {
      oneToks = stripPunctuations(stripDeterminers(oneToks));
      twoToks = stripPunctuations(stripDeterminers(twoToks));
      if(Utils.levenshteinDistance(oneToks, twoToks) == 1) {
        if(oneToks.length == twoToks.length) {
          Iterator<String> iter1 = Arrays.asList(oneToks).iterator();
          Iterator<String> iter2 = Arrays.asList(twoToks).iterator();
          String str1 = popNextOrNull(iter1);
          String str2 = popNextOrNull(iter2);
          // find and match acronyms for different token
          while(str1 != null && str2 != null) {
            if(!str1.equals(str2)) {
              if((abbreviations.containsKey(str1.toLowerCase()) && abbreviations.get(str1.toLowerCase()).equalsIgnoreCase(str2)) || (abbreviations.containsKey(str2.toLowerCase()) && abbreviations.get(str2.toLowerCase()).equalsIgnoreCase(str1))) {
                //logger.log(featureType+":NameAcronym",name1+"\t"+name2);
                return true;
              }
            }
            str1 = popNextOrNull(iter1);
            str2 = popNextOrNull(iter2);
          }
        } else {
          return false;
        }
      }
      return false;
  }
  }

  public static <X> X popNextOrNull(Iterator<X> p) {
    if (p.hasNext()) {
      return p.next();
    } else {
      return null;
    }
  }

  private static <E> Set<E> mkSet(E[] array){
                Set<E> rtn = new HashSet<E>();
                Collections.addAll(rtn, array);
                return rtn;
        }

  private final Set<Object> FEATURES = mkSet(new Object[] {
          // Conditional features look like:
          new ConditionalFeature.Specification(Arrays.asList("all", "ner", "head", "none"), Feature.ExactMatch.class),
          new ConditionalFeature.Specification(Arrays.asList("ner"), Feature.TokenEditDistance.class),
          new ConditionalFeature.Specification(Arrays.asList("ner", "head"), Feature.EditDistance.class),
          new ConditionalFeature.Specification(Arrays.asList("ner", "none"), Feature.NameAcronym.class),
          new ConditionalFeature.Specification(Arrays.asList("all", "ner"), Feature.DifferentBy.class),
          new ConditionalFeature.Specification(Arrays.asList("all"), Feature.MatchNounTokens.class),
          new ConditionalFeature.Specification(Arrays.asList("all"), Feature.MatchVerbTokens.class),
          new ConditionalFeature.Specification(Arrays.asList("all"), Feature.NickNameMatch.class),
          // ...
          // Put in all your features here
          /*Feature.ExactMatch.class,
          Feature.HeadMatch.class,
          Feature.FirstNameMatch.class,
          Feature.LastNameMatch.class,
          Feature.NickNameMatch.class,
          Feature.EditDistance.class,
          Feature.EntityLengthDiff.class,
          Feature.NameNERMatch.class,
          Feature.NameAcronym.class,
          Feature.TokenEditDistance.class,
          Feature.FuzzyNameMatch.class,
          Feature.DifferentBy.class,
          Feature.FirstNameEditDistance.class,
          Feature.MiddleNameEditDistance.class,
          Feature.MatchNounTokens.class,
          Feature.MatchVerbTokens.class,*/
          //Feature.ContextLengthDiff.class,
          //Feature.NERAnyMatch.class,
          //Feature.NERTotalMatch.class,
          //Feature.PrevNER.class,
          //Feature.NextNER.class,
          //Feature.POSTotalMatch.class,
          //Feature.NextPOS.class,
          //Feature.PrevPOS.class,
          //Feature.MatchTokens.class,
          //Feature.SentenceAcronym.class,

  });


  public EntityLinkingFeaturizer() {
    // Load names
    maleNamesLowerCase = Collections.unmodifiableMap(readNicknames(Props.ENTITYLINKING_MALENAMES.getPath()));
    femaleNamesLowerCase = Collections.unmodifiableMap(readNicknames(Props.ENTITYLINKING_FEMALENAMES.getPath()));
    abbreviations = Collections.unmodifiableMap(readAbbreviations(Props.ENTITYLINKING_ABBREVIATIONS.getPath()));
    countsFeature = new int[FEATURES.size()];
  }


  /**
   * TODO(melvin) implement me!
   * TODO(melvin) document me!
   *
   *
   * @param featureClass The feature class to extract.
   * @param gloss The gloss, <b>already clipped to the appropriate span (head, NER, all, etc.)</b>
   * @param tokens The tokens, likewise already clipped.
   * @param tokensInfo The actual CoreLabels for the otkens, likewise already clipped.
   * @param context The context. Note that the span in this context is of the clipped span -- not the original span.
   * @param tags
   * @param featureType
   * @return A feature of the given class according to the given data.
   */
  private <E> Feature featurize(Class<E> featureClass,
                                Pair<String, String> gloss,
                                Pair<String[], String[]> tokens,
                                Maybe<Pair<List<CoreLabel>, List<CoreLabel>>> tokensInfo,
                                Maybe<Pair<Pair<CoreMap, Span>, Pair<CoreMap, Span>>> context, Pair<NERTag, NERTag> tags, String featureType) {
    // -- Features over the surface form
    String name1 = gloss.first;
    String name2 = gloss.second;
    // TODO(melvin)
    if(featureClass.equals(Feature.ExactMatch.class)) {
      if(name1.equalsIgnoreCase(name2)) {
        //logger.log(featureType+":ExactMatch",name1+"\t"+name2);
        return new Feature.ExactMatch(true);
      } else {
        return null;
      }
    } else if(featureClass.equals(Feature.EditDistance.class)) {
      /*int bucket = bucketScore(JaroWinkler.similarity(name1, name2));
      if(bucket ==0) {
        return null;
      }
      return new Feature.EditDistance(bucket);*/
      int editDist = Utils.levenshteinDistance(name1, name2);
      if(editDist == 0) {
        return null;
      }
      if(editDist <=3) {
        //logger.log(featureType+":EditDist("+editDist+")",name1+"\t"+name2);
        return new Feature.EditDistance(editDist);
      } else {
        //logger.log(featureType+":EditDist(4)",name1+"\t"+name2);
        return new Feature.EditDistance(4);
      }
    }
    // -- Features over the tokens
    String[] oneToks = tokens.first;
    String[] twoToks = tokens.second;
    // TODO(melvin)
    if(featureClass.equals(Feature.NameAcronym.class)) {
      if(AcronymMatcher.isAcronym(oneToks, twoToks)) {
        //countsFeature[8]++;
        return new Feature.NameAcronym(true);
      } else {
        oneToks = stripPunctuations(stripDeterminers(oneToks));
        twoToks = stripPunctuations(stripDeterminers(twoToks));
        if(Utils.levenshteinDistance(oneToks, twoToks) == 1) {
          if(oneToks.length == twoToks.length) {
            Iterator<String> iter1 = Arrays.asList(oneToks).iterator();
            Iterator<String> iter2 = Arrays.asList(twoToks).iterator();
            String str1 = popNextOrNull(iter1);
            String str2 = popNextOrNull(iter2);
            // find and match acronyms for different token
            while(str1 != null && str2 != null) {
              if(!str1.equals(str2)) {
                if((abbreviations.containsKey(str1.toLowerCase()) && abbreviations.get(str1.toLowerCase()).equalsIgnoreCase(str2)) || (abbreviations.containsKey(str2.toLowerCase()) && abbreviations.get(str2.toLowerCase()).equalsIgnoreCase(str1))) {
                  //logger.log(featureType+":NameAcronym",name1+"\t"+name2);
                  return new Feature.NameAcronym(true);
                }
              }
              str1 = popNextOrNull(iter1);
              str2 = popNextOrNull(iter2);
            }
          } else {
            return null;
          }
        }
        return null;
      }
    } else if(featureClass.equals(Feature.TokenEditDistance.class)) {
      double longer = oneToks.length > twoToks.length ? oneToks.length : twoToks.length;
      int editDist = Utils.levenshteinDistance(oneToks, twoToks);
      if(editDist == 0) {
        return null;
      }
      if(editDist <=3) {
        //logger.log(featureType+":TokenED("+editDist+")",name1+"\t"+name2);
        if(isAcronym(oneToks, twoToks)) {
          return new Feature.TokenEditDistance(editDist);
        } else {
          return null;
        }
      } else {
        //logger.log(featureType+":TokenED(4)",name1+"\t"+name2);
        if(isAcronym(oneToks, twoToks)) {
          return new Feature.TokenEditDistance(4);
        } else {
          return null;
        }
      }
    } else if(featureClass.equals(Feature.NickNameMatch.class)) {
      // Nicknames
      if (tags.first == NERTag.PERSON && tags.second == NERTag.PERSON && oneToks.length >= 2 && twoToks.length >= 2 &&
              oneToks[oneToks.length - 1].toLowerCase().equals(twoToks[twoToks.length - 1].toLowerCase())) {
        // case: last names match
        String firstNameOne = oneToks[0].toLowerCase();
        String firstNameTwo = twoToks[0].toLowerCase();
        //noinspection StringEquality  // Safe, as the Strings are canonicalized; != allows for more elegant handling of null,StringEquality
        if ((maleNamesLowerCase.containsKey(firstNameOne) && maleNamesLowerCase.get(firstNameOne).equals(maleNamesLowerCase.get(firstNameTwo))) ||
                (femaleNamesLowerCase.containsKey(firstNameOne) && femaleNamesLowerCase.get(firstNameOne).equals(femaleNamesLowerCase.get(firstNameTwo)))) {
          // First names are nicknames of each other -- they're likely the same people.
          //logger.log(featureType+":NickNameMatch",name1+"\t"+name2);
          return new Feature.NickNameMatch(true);
        }
      }
      return null;
    }

    // -- Features over the Token Info
    for (Pair<List<CoreLabel>, List<CoreLabel>> tokenInfoImpl : tokensInfo) {
      List<CoreLabel> tokenInfo1 = tokenInfoImpl.first;
      List<CoreLabel> tokenInfo2 = tokenInfoImpl.second;
      // TODO(melvin)
      if(featureClass.equals(Feature.DifferentBy.class)) {
        if(Utils.levenshteinDistance(oneToks, twoToks) == 1) {
          Iterator<CoreLabel> iter1 = tokenInfo1.iterator();
          Iterator<CoreLabel> iter2 = tokenInfo2.iterator();
          CoreLabel core1 = popNextOrNull(iter1);
          CoreLabel core2 = popNextOrNull(iter2);
          // find and return different token
          while(core1 != null && core2 != null) {
            if(core1.word().equals(core2.word())) {
              core1 = popNextOrNull(iter1);
              core2 = popNextOrNull(iter2);
            } else {
              if(tokenInfo1.size() > tokenInfo2.size()) {
                if(CORPORATE_SUFFIXES.contains(core1.word().toLowerCase())) {
                  //logger.log(featureType+":DiffBy(Diff-Corp)",name1+"\t"+name2);
                  return new Feature.DifferentBy("Diff-Corp");
                } else {
                  if(core1.tag().startsWith("N") && !core1.tag().startsWith("NNP")) {
                    //logger.log(featureType+":DiffBy(Diff-N)",name1+"\t"+name2);
                    return new Feature.DifferentBy("Diff-N");
                  } else if(core1.tag().startsWith("N")){
                    //logger.log(featureType+":DiffBy(Diff-PN)",name1+"\t"+name2);
                    return new Feature.DifferentBy("Diff-PN");
                  } else {
                    if(core1.tag().charAt(0) == ':' || core1.tag().charAt(0) == ',' || core1.tag().charAt(0) == ',' || core1.tag().charAt(0) == '.' || core1.tag().charAt(0) == '\'') {
                      //logger.log(featureType+":DiffBy(Diff-Punc)",name1+"\t"+name2);
                      return new Feature.DifferentBy("Diff-Punc");
                    }
                    //logger.log(featureType+":DiffBy("+core1.tag().charAt(0)+")",name1+"\t"+name2);
                    return new Feature.DifferentBy("Diff-"+core1.tag().charAt(0));
                  }
                }
              } else {
                if(CORPORATE_SUFFIXES.contains(core2.word().toLowerCase())) {
                  //logger.log(featureType+":DiffBy(Diff-Corp)",name1+"\t"+name2);
                  return new Feature.DifferentBy("Diff-Corp");
                } else {
                  if(core2.tag().startsWith("N") && !core2.tag().startsWith("NNP")) {
                    //logger.log(featureType+":DiffBy(Diff-N)",name1+"\t"+name2);
                    return new Feature.DifferentBy("Diff-N");
                  } else if(core2.tag().startsWith("N")){
                    //logger.log(featureType+":DiffBy(Diff-NP)",name1+"\t"+name2);
                    return new Feature.DifferentBy("Diff-NP");
                  } else {
                    if(core2.tag().charAt(0) == ':' || core2.tag().charAt(0) == ',' || core2.tag().charAt(0) == ',' || core2.tag().charAt(0) == '.' || core2.tag().charAt(0) == '\'') {
                      //logger.log(featureType+":DiffBy(Diff-Punc)",name1+"\t"+name2);
                      return new Feature.DifferentBy("Diff-Punc");
                    }
                    //logger.log(featureType+":DiffBy("+core2.tag().charAt(0)+")",name1+"\t"+name2);
                    return new Feature.DifferentBy("Diff-"+core2.tag().charAt(0));
                  }
                }
              }
            }
          }
          if(core1 != null) {
            if(CORPORATE_SUFFIXES.contains(core1.word().toLowerCase())) {
              //logger.log(featureType+":DiffBy(Diff-Corp)",name1+"\t"+name2);
              return new Feature.DifferentBy("Diff-Corp");
            } else {
              if(core1.tag().startsWith("N") && !core1.tag().startsWith("NNP")) {
                //logger.log(featureType+":DiffBy(Diff-N)",name1+"\t"+name2);
                return new Feature.DifferentBy("Diff-N");
              } else if(core1.tag().startsWith("N")){
                //logger.log(featureType+":DiffBy(Diff-PN)",name1+"\t"+name2);
                return new Feature.DifferentBy("Diff-PN");
              } else {
                if(core1.tag().charAt(0) == ':' || core1.tag().charAt(0) == ',' || core1.tag().charAt(0) == ',' || core1.tag().charAt(0) == '.' || core1.tag().charAt(0) == '\'') {
                  //logger.log(featureType+":DiffBy(Diff-Punc)",name1+"\t"+name2);
                  return new Feature.DifferentBy("Diff-Punc");
                }
                //logger.log(featureType+":DiffBy("+core1.tag().charAt(0)+")",name1+"\t"+name2);
                return new Feature.DifferentBy("Diff-"+core1.tag().charAt(0));
              }
            }
          }
          if(core2 != null) {
            if(CORPORATE_SUFFIXES.contains(core2.word().toLowerCase())) {
              //logger.log(featureType+":DiffBy(Diff-Corp)",name1+"\t"+name2);
              return new Feature.DifferentBy("Diff-Corp");
            } else {
              if(core2.tag().startsWith("N") && !core2.tag().startsWith("NNP")) {
                //logger.log(featureType+":DiffBy(Diff-N)",name1+"\t"+name2);
                return new Feature.DifferentBy("Diff-N");
              } else if(core2.tag().startsWith("N")){
                //logger.log(featureType+":DiffBy(Diff-NP)",name1+"\t"+name2);
                return new Feature.DifferentBy("Diff-NP");
              } else {
                if(core2.tag().charAt(0) == ':' || core2.tag().charAt(0) == ',' || core2.tag().charAt(0) == ',' || core2.tag().charAt(0) == '.' || core2.tag().charAt(0) == '\'') {
                  //logger.log(featureType+":DiffBy(Diff-Punc)",name1+"\t"+name2);
                  return new Feature.DifferentBy("Diff-Punc");
                }
                //logger.log(featureType+":DiffBy("+core2.tag().charAt(0)+")",name1+"\t"+name2);
                return new Feature.DifferentBy("Diff-"+core2.tag().charAt(0));
              }
            }
          }
        } else {
          return null;
        }
      }
      // -- Features over the context
      for (Pair<Pair<CoreMap, Span>, Pair<CoreMap, Span>> contextImpl : context) {
        CoreMap sentence1 = contextImpl.first.first;
        Span entitySpan1  = contextImpl.first.second;
        CoreMap sentence2 = contextImpl.second.first;
        Span entitySpan2  = contextImpl.second.second;
        // TODO(melvin)
        if(featureClass.equals(Feature.MatchNounTokens.class)) {

          // get Corelabels and find tokens
          List<CoreLabel> oneLabels = sentence1.get(CoreAnnotations.TokensAnnotation.class);
          List<CoreLabel> twoLabels = sentence2.get(CoreAnnotations.TokensAnnotation.class);

          int match = 0;
          // Get number of matching tokens between the two entity sentences
          boolean[] matchedHigherToks = new boolean[oneLabels.size()];
          boolean[] matchedLowerToks = new boolean[twoLabels.size()];
          int count1 = 0;
          int count2 = 0;
          for (int h = 0; h < oneLabels.size(); ++h) {
            if (matchedHigherToks[h]) { continue; }
            String higherTok = oneLabels.get(h).word();
            String higherTokNoSpecialChars = Utils.noSpecialChars(higherTok);
            String higherPos = oneLabels.get(h).tag();
            if(higherPos.startsWith("N")) {
              count1++;
            }
            boolean doesMatch = false;
            for (int l = 0; l < twoLabels.size(); ++l) {
              if (matchedLowerToks[l]) { continue; }
              String lowerTok = twoLabels.get(l).word();
              String lowerTokNoSpecialCars = Utils.noSpecialChars(lowerTok);
              String lowerPos = twoLabels.get(l).tag();
              if(lowerPos.startsWith("N") && h==0) {
                count2++;
              }
              if (higherTokNoSpecialChars.equalsIgnoreCase(lowerTokNoSpecialCars) && (higherPos.startsWith("N") && lowerPos.startsWith("N"))) {
                doesMatch = true;
                matchedHigherToks[h] = true;
                matchedLowerToks[l] = true;
              }
            }
            if (doesMatch) { match += 1; }
          }
          if(match ==0) {
            return null;
          } else if(match <=3) {
            //logger.log(featureType+":MatchNT("+match+")",name1+"\t"+name2);
            return new Feature.MatchNounTokens(match);
          } else {
            //logger.log(featureType+":MatchNT(4)",name1+"\t"+name2);
            return new Feature.MatchNounTokens(4);
          }
        } else if(featureClass.equals(Feature.MatchVerbTokens.class)) {
          // get Corelabels and find tokens

          List<CoreLabel> oneLabels = sentence1.get(CoreAnnotations.TokensAnnotation.class);
          List<CoreLabel> twoLabels = sentence2.get(CoreAnnotations.TokensAnnotation.class);

          int match = 0;
          // Get number of matching tokens between the two entity sentences
          boolean[] matchedHigherToks = new boolean[oneLabels.size()];
          boolean[] matchedLowerToks = new boolean[twoLabels.size()];
          int count1 = 0;
          int count2 = 0;
          for (int h = 0; h < oneLabels.size(); ++h) {
            if (matchedHigherToks[h]) { continue; }
            String higherTok = oneLabels.get(h).word();
            String higherTokNoSpecialChars = Utils.noSpecialChars(higherTok);
            String higherPos = oneLabels.get(h).tag();
            if(higherPos.startsWith("V")) {
              count1++;
            }
            boolean doesMatch = false;
            for (int l = 0; l < twoLabels.size(); ++l) {
              if (matchedLowerToks[l]) { continue; }
              String lowerTok = twoLabels.get(l).word();
              String lowerTokNoSpecialCars = Utils.noSpecialChars(lowerTok);
              String lowerPos = twoLabels.get(l).tag();
              if(lowerPos.startsWith("V") && h==0) {
                count2++;
              }
              if (higherTokNoSpecialChars.equalsIgnoreCase(lowerTokNoSpecialCars) && (higherPos.startsWith("V") && lowerPos.startsWith("V"))) {
                doesMatch = true;
                matchedHigherToks[h] = true;
                matchedLowerToks[l] = true;
              }
            }
            if (doesMatch) { match += 1; }
          }
          if(match ==0) {
            return null;
          } else if(match <=3) {
            //logger.log(featureType+":MatchVT("+match+")",name1+"\t"+name2);
            return new Feature.MatchVerbTokens(match);
          } else {
            //logger.log(featureType+":MatchVT(4)",name1+"\t"+name2);
            return new Feature.MatchVerbTokens(4);
          }
        }
      }
    }

    // -- Could not find feature implementation
    logger.debug("Feature implementation not defined for " + featureClass);
    return null;
  }

  private int bucketScore(double score) {
    if(score < 0.75) {
      return 1;
    } else if (score >= 0.75 && score < 0.80) {
      return 2;
    } else if(score >= 0.80 && score < 0.85) {
      return 3;
    } else if(score >= 0.85 && score < 0.90) {
      return 4;
    } else if(score >= 0.90 && score < 0.95) {
      return 5;
    } else if(score >= 0.95 && score < 1) {
      return 6;
    } else {
      return 0;
    }
  }

  /**
   * Get the actual list of core labels corresponding to this entity context's entity.
   * If the context does not provide this information, return {@link edu.stanford.nlp.kbp.common.Maybe.Nothing}.
   * @param context The entity context to get the span for.
   * @return The span of <b>only the entity</b> within the sentence, if defined.
   */
  private Maybe<List<CoreLabel>> entitySpanInSentence(EntityContext context) {
    if (context.document.isDefined() && context.sentenceIndex.isDefined() && context.entityTokenSpan.isDefined()) {
      return Maybe.Just(context.document.get().get(CoreAnnotations.SentencesAnnotation.class).get(context.sentenceIndex.get()).get(CoreAnnotations.TokensAnnotation.class).subList(context.entityTokenSpan.get().start(), context.entityTokenSpan.get().end()));
    } else if (context.sentence.isDefined() && context.entityTokenSpan.isDefined()) {
      return Maybe.Just(context.sentence.get().get(CoreAnnotations.TokensAnnotation.class).subList(context.entityTokenSpan.get().start(), context.entityTokenSpan.get().end()));
    } else {
      return Maybe.Nothing();
    }
  }

  /**
   * Do a heuristic head find on the entity context's containing sentence, if it's defined.
   */
  private Maybe<Span> headSpanInSentence(EntityContext context) {
    // Get the tree
    Tree tree;
    if (context.document.isDefined() && context.sentenceIndex.isDefined() && context.entityTokenSpan.isDefined()) {
      tree = context.document.get().get(CoreAnnotations.SentencesAnnotation.class).get(context.sentenceIndex.get()).get(TreeCoreAnnotations.TreeAnnotation.class);
    } else if (context.sentence.isDefined() && context.entityTokenSpan.isDefined()) {
      tree = context.sentence.get().get(TreeCoreAnnotations.TreeAnnotation.class);
    } else {
      return Maybe.Nothing();
    }
    // Compute the spans in the tree
    tree.setSpans();
    if (context.entityTokenSpan.isDefined()) {
      // Get the best matching subtree for this entity context
      Span entityTokenSpan = context.entityTokenSpan.get();
      int min = Integer.MAX_VALUE;
      Tree argmin = null;
      for (Tree subtree : tree) {
        IntPair treeSpan = subtree.getSpan();
        int candMin = Math.abs((treeSpan.getSource()) - entityTokenSpan.start()) + Math.abs((treeSpan.getTarget()) - entityTokenSpan.end());
        if (candMin < min) {
          min = candMin;
          argmin = subtree;
        }
      }
      if (argmin != null) {
        // Find it's head
        Tree head = headFinder.determineHead(argmin);
        if (head != null) {
          // Return the span for the head
          return Maybe.Just(new Span(head.getSpan().getSource() , head.getSpan().getTarget() +1));
        } else {
          // Case: no head found; backoff to last token in span
          return Maybe.Just(new Span(entityTokenSpan.end() -1, entityTokenSpan.end()));
        }
      } else {
        // Case: no good span match found; backoff to last token in span
        return Maybe.Just(new Span(entityTokenSpan.end() -1, entityTokenSpan.end()));
      }
    } else {
      // Case: no sentence found. We're out of luck.
      return Maybe.Nothing();
    }
  }

  /**
   * TODO(melvin) documentation
   * @param featureSpec
   * @param input
   * @param <E>
   * @return
   */
  protected <E> Feature featurize2(Object featureSpec, Pair<EntityContext, EntityContext> input) {
    // Parameterize the feature spec
    Class<? extends Feature> featureClass = null;
    List<String> featureTypeList = new ArrayList<>();
    if (featureSpec instanceof ConditionalFeature.Specification) {
      featureClass = ((ConditionalFeature.Specification) featureSpec).featureClass;
      featureTypeList = ((ConditionalFeature.Specification) featureSpec).conditions;
    } else if (featureSpec instanceof Class) {
      //noinspection unchecked
      featureClass = (Class) featureSpec;
    } else {
      throw new IllegalArgumentException("Unknown feature specifier: " + featureSpec);
    }
    for(int i=0; i< featureTypeList.size(); i++) {
      String featureType = featureTypeList.get(i);

      // check condition for "none"
      if(featureType.equals("none")) {
        return new ConditionalFeature(featureType, featurizeDummy(featureClass));
      }
      // Route feature
      // (get spans according to routing)
      Maybe<List<CoreLabel>> entitySpan1 = entitySpanInSentence(input.first);
      Maybe<List<CoreLabel>> entitySpan2 = entitySpanInSentence(input.second);
      switch (featureType) {
        case "all":
          break;
        case "ner":
          for (List<CoreLabel> span : entitySpan1) {
            NERTag type = input.first.entity.type;
            while (span.size() > 1 && !span.get(0).ner().equals(type.name)) {
              span = span.subList(1, span.size());
            }
            while (span.size() > 1 && !span.get(span.size() - 1).ner().equals(type.name)) {
              span = span.subList(0, span.size() - 1);
            }
            entitySpan1 = Maybe.Just(span);
          }
          for (List<CoreLabel> span : entitySpan2) {
            NERTag type = input.second.entity.type;
            while (span.size() > 1 && !span.get(0).ner().equals(type.name)) {
              span = span.subList(1, span.size());
            }
            while (span.size() > 1 && !span.get(span.size() - 1).ner().equals(type.name)) {
              span = span.subList(0, span.size() - 1);
            }
            entitySpan2 = Maybe.Just(span);
          }
          break;
        case "head":
          for (Span headSpan : headSpanInSentence(input.first)) {
            EntityContext context = input.first;
            if (context.document.isDefined() && context.sentenceIndex.isDefined() && context.entityTokenSpan.isDefined()) {
              entitySpan1 = Maybe.Just(context.document.get().get(CoreAnnotations.SentencesAnnotation.class).get(context.sentenceIndex.get()).get(CoreAnnotations.TokensAnnotation.class).subList(headSpan.start(), headSpan.end()));
            } else if (context.sentence.isDefined() && context.entityTokenSpan.isDefined()) {
              entitySpan1 = Maybe.Just(context.sentence.get().get(CoreAnnotations.TokensAnnotation.class).subList(headSpan.start(), headSpan.end()));
            }
          }
          for (Span headSpan : headSpanInSentence(input.second)) {
            EntityContext context = input.second;
            if (context.document.isDefined() && context.sentenceIndex.isDefined() && context.entityTokenSpan.isDefined()) {
              entitySpan2 = Maybe.Just(context.document.get().get(CoreAnnotations.SentencesAnnotation.class).get(context.sentenceIndex.get()).get(CoreAnnotations.TokensAnnotation.class).subList(headSpan.start(), headSpan.end()));
            } else if (context.sentence.isDefined() && context.entityTokenSpan.isDefined()) {
              entitySpan2 = Maybe.Just(context.sentence.get().get(CoreAnnotations.TokensAnnotation.class).subList(headSpan.start(), headSpan.end()));
            }
          }
          break;
      }
      // (gloss)
      Pair<String, String> gloss;
      if (entitySpan1.isDefined() && entitySpan2.isDefined()) {
        gloss = Pair.makePair(CoreMapUtils.phraseToOriginalString(entitySpan1.get()), CoreMapUtils.phraseToOriginalString(entitySpan2.get()));
      } else {
        if(featureType.equals("all")) {
          gloss = Pair.makePair(input.first.entity.name, input.second.entity.name);
        } else {
          continue;
        }
      }
      // (tokens)
      Pair<String[], String[]> tokens;
      if (entitySpan1.isDefined() && entitySpan2.isDefined()) {
        tokens = Pair.makePair(CoreMapUtils.phraseToOriginalTokens(entitySpan1.get()), CoreMapUtils.phraseToOriginalTokens(entitySpan2.get()));
      } else {
        if(featureType.equals("all")) {
          tokens = Pair.makePair(input.first.entity.name.split("\\s+"), input.second.entity.name.split("\\s+"));
        } else {
          continue;
        }
      }
      // (token info)
      Maybe<Pair<List<CoreLabel>, List<CoreLabel>>> tokensInfo = Maybe.Nothing();
      if (entitySpan1.isDefined() && entitySpan2.isDefined()) {
        tokensInfo = Maybe.Just(Pair.makePair(entitySpan1.get(), entitySpan2.get()));
      }
      // (context info)
      Maybe<Pair<Pair<CoreMap,Span>, Pair<CoreMap,Span>>> context = Maybe.Nothing();
      Maybe<CoreMap> entity1Sentence = input.first.sentence;
      if (input.first.document.isDefined() && input.first.sentenceIndex.isDefined()) {
        entity1Sentence = Maybe.Just(input.first.document.get().get(CoreAnnotations.SentencesAnnotation.class).get(input.first.sentenceIndex.get()));
      }
      Maybe<CoreMap> entity2Sentence = input.second.sentence;
      if (input.second.document.isDefined() && input.second.sentenceIndex.isDefined()) {
        entity2Sentence = Maybe.Just(input.second.document.get().get(CoreAnnotations.SentencesAnnotation.class).get(input.second.sentenceIndex.get()));
      }
      for (CoreMap e1Sentence : entity1Sentence) {
        for (CoreMap e2Sentence : entity2Sentence) {
          for (List<CoreLabel> e1Span : entitySpan1) {
            for (List<CoreLabel> e2Span : entitySpan2) {
              Span e1SpanAsSpan = new Span(e1Span.get(0).index(), e1Span.get(e1Span.size() - 1).index() + 1);
              Span e2SpanAsSpan = new Span(e2Span.get(0).index(), e2Span.get(e2Span.size() - 1).index() + 1);
              context = Maybe.Just(Pair.makePair(Pair.makePair(e1Sentence, e1SpanAsSpan), Pair.makePair(e2Sentence, e2SpanAsSpan)));
            }
          }
        }
      }
      // ner tags
      Pair<NERTag, NERTag> tags = Pair.makePair(input.first.entity.type, input.second.entity.type);

      // Run featurizer
      Feature feature = featurize(featureClass, gloss, tokens, tokensInfo, context, tags, featureType);
      if (featureSpec instanceof ConditionalFeature.Specification) {
        if(feature == null) {
          continue;
        }
        return new ConditionalFeature(featureType, feature);
      } else {
        return feature;
      }
    }
    return null;
  }

  private Feature featurizeDummy(Class<? extends Feature> featureClass) {
    if(featureClass.equals(Feature.ExactMatch.class)) {
      return new Feature.ExactMatch(true);
    } else if(featureClass.equals(Feature.NameAcronym.class)) {
      return new Feature.NameAcronym(true);
    } else {
      return null;
    }
  }


  /**
   * The implementation of the featurize function.
   *
   * The features should attempt to encode the affinity for these to entities for each other;
   * additional resources needed to make that judgment can be passed in through the featurizer's
   * constructor, or added to the entity context if appropriate.
   * When choosing the second option, however, be wary that any information put into the entity context
   * must be populated at test time as well, from whatever context the entity context is created.
   */
  /*
  Feature function
   */
  private <E> Feature feature(Class<E> clazz, Pair<EntityContext, EntityContext> input ){

    EntityContext entityOne = input.first;
    EntityContext entityTwo = input.second;
    String name1 = entityOne.entity.name;
    String name2 = entityTwo.entity.name;
    NERTag ner1 = entityOne.entity.type;
    NERTag ner2 = entityTwo.entity.type;
    Maybe<Span> span1 = entityOne.entityTokenSpan;
    Maybe<Span> span2 = entityTwo.entityTokenSpan;
    Maybe<CoreMap> sentence1 = entityOne.sentence;
    Maybe<CoreMap> sentence2 = entityTwo.sentence;

    if(clazz.equals(Feature.ExactMatch.class)){
      if(name1.equals(name2)) {
        countsFeature[0]++;
        return new Feature.ExactMatch(true);
      } else {
        return new Feature.ExactMatch(false);
      }
    } else if(clazz.equals(Feature.HeadMatch.class)) {
      name1 = stripDeterminers(name1);
      name2 = stripDeterminers(name2);
      if(name1.split("\\s+")[0].equalsIgnoreCase(name2.split("\\s+")[0])) {
        countsFeature[1]++;
        return new Feature.HeadMatch(true);
      } else {
        return new Feature.HeadMatch(false);
      }
    } else if(clazz.equals(Feature.LastNameMatch.class)) {
      if (ner1 == NERTag.PERSON && ner2 == NERTag.PERSON && (entityOne.tokens().length >= 2 || entityTwo.tokens().length >= 2) ) {
        // case: last names match
        if(entityOne.tokens()[entityOne.tokens().length - 1].toLowerCase().equalsIgnoreCase(entityTwo.tokens()[entityTwo.tokens().length - 1].toLowerCase())) {
          countsFeature[3]++;
          return new Feature.LastNameMatch(true);
        } else {
          return new Feature.LastNameMatch(false);
        }
      }
      return new Feature.LastNameMatch(false);
    } else if(clazz.equals(Feature.FirstNameMatch.class)) {
      if (ner1 == NERTag.PERSON && ner2 == NERTag.PERSON && (entityOne.tokens().length >= 2 || entityTwo.tokens().length >= 2) ) {
        // case: first names match
        if(entityOne.tokens()[0].toLowerCase().equalsIgnoreCase(entityTwo.tokens()[0].toLowerCase())) {
          countsFeature[2]++;
          return new Feature.FirstNameMatch(true);
        } else {
          return new Feature.FirstNameMatch(false);
        }
      }
      return new Feature.FirstNameMatch(false);
    } else if(clazz.equals(Feature.FirstNameEditDistance.class)) {
      if (ner1 == NERTag.PERSON && ner2 == NERTag.PERSON && (entityOne.tokens().length >= 2 || entityTwo.tokens().length >= 2) ) {
        // case: last names match
        if(entityOne.tokens()[entityOne.tokens().length - 1].toLowerCase().equalsIgnoreCase(entityTwo.tokens()[entityTwo.tokens().length - 1].toLowerCase())) {
          String firstName1 = entityOne.tokens()[0];
          String firstName2 = entityTwo.tokens()[0];
          double longer = firstName1.length() > firstName2.length() ? firstName1.length() : firstName2.length();
          countsFeature[12]++;
          return new Feature.FirstNameEditDistance(Utils.levenshteinDistance(firstName1, firstName2) / longer);
        }
      }
      return null;
    } else if(clazz.equals(Feature.MiddleNameEditDistance.class)) {
      if (ner1 == NERTag.PERSON && ner2 == NERTag.PERSON && (entityOne.tokens().length >= 2 || entityTwo.tokens().length >= 2) ) {
        // case: last names match and first names match
        if((entityOne.tokens()[entityOne.tokens().length - 1].toLowerCase().equalsIgnoreCase(entityTwo.tokens()[entityTwo.tokens().length - 1].toLowerCase())) && (entityOne.tokens()[0].toLowerCase().equalsIgnoreCase(entityTwo.tokens()[0].toLowerCase()))) {
          String middleName1 = StringUtils.join(entityOne.tokens(), ' ', 1, entityOne.tokens().length - 1);
          String middleName2 = StringUtils.join(entityTwo.tokens(), ' ', 1, entityTwo.tokens().length-1);
          double longer = middleName1.length() > middleName2.length() ? middleName1.length() : middleName2.length();
          countsFeature[13]++;
          return new Feature.MiddleNameEditDistance(Utils.levenshteinDistance(middleName1, middleName2) / longer);
        }
      }
      return null;
    } else if(clazz.equals(Feature.NickNameMatch.class)) {
      // Nicknames
      if (ner1 == NERTag.PERSON && ner2 == NERTag.PERSON && entityOne.tokens().length >= 2 && entityTwo.tokens().length >= 2 &&
              entityOne.tokens()[entityOne.tokens().length - 1].toLowerCase().equals(entityTwo.tokens()[entityTwo.tokens().length - 1].toLowerCase())) {
        // case: last names match
        String firstNameOne = entityOne.tokens()[0].toLowerCase();
        String firstNameTwo = entityTwo.tokens()[0].toLowerCase();
        //noinspection StringEquality  // Safe, as the Strings are canonicalized; != allows for more elegant handling of null,StringEquality
        if ((maleNamesLowerCase.containsKey(firstNameOne) && maleNamesLowerCase.get(firstNameOne).equals(maleNamesLowerCase.get(firstNameTwo))) ||
                (femaleNamesLowerCase.containsKey(firstNameOne) && femaleNamesLowerCase.get(firstNameOne).equals(femaleNamesLowerCase.get(firstNameTwo)))) {
          // First names are nicknames of each other -- they're likely the same people.
          countsFeature[4]++;
          return new Feature.NickNameMatch(true);
        }
      }
      return new Feature.NickNameMatch(false);
    } else if(clazz.equals(Feature.EditDistance.class)) {
      double longer = name1.length() > name2.length() ? name1.length() : name2.length();
      countsFeature[5]++;
      return new Feature.EditDistance((int) (Utils.levenshteinDistance(name1, name2)/ longer));
    } else if(clazz.equals(Feature.TokenEditDistance.class)) {
      String[] oneToks = entityOne.tokens();
      String[] twoToks = entityTwo.tokens();
      double longer = oneToks.length > twoToks.length ? oneToks.length : twoToks.length;
      countsFeature[9]++;
      return new Feature.TokenEditDistance((int) (Utils.levenshteinDistance(oneToks, twoToks)/ longer));
    } else if(clazz.equals(Feature.FuzzyNameMatch.class)) {
      if(stripDeterminers(stripCorporateTitles(name1)).equalsIgnoreCase(stripDeterminers(stripCorporateTitles(name2)))) {
        countsFeature[10]++;
        return new Feature.FuzzyNameMatch(true);
      } else {
        return new Feature.FuzzyNameMatch(false);
      }
    } else if(clazz.equals(Feature.DifferentBy.class)) {
      String[] oneToks = entityOne.tokens();
      String[] twoToks = entityTwo.tokens();
      if(Utils.levenshteinDistance(oneToks, twoToks) == 1) {
        Iterator<String> iter1 = Arrays.asList(oneToks).iterator();
        Iterator<String> iter2 = Arrays.asList(twoToks).iterator();
        String str1 = popNextOrNull(iter1);
        String str2 = popNextOrNull(iter2);
        countsFeature[11]++;
        // find and return different token
        while(str1 != null && str2 != null) {
          if(str1.equals(str2)) {
            str1 = popNextOrNull(iter1);
            str2 = popNextOrNull(iter2);
          } else {
            if(oneToks.length > twoToks.length) {
              return new Feature.DifferentBy(str1);
            } else {
              return new Feature.DifferentBy(str2);
            }
          }
        }
        if(str1 != null) {
          return new Feature.DifferentBy(str1);
        } else {
          return new Feature.DifferentBy(str2);
        }
      } else {
        return null;
      }
    } else if(clazz.equals(Feature.PrevPOS.class)) {
      if(span1.isDefined() && span2.isDefined() && sentence1.isDefined() && sentence2.isDefined()) {
        if(span1.get().start()-1 >= 0 && span2.get().start()-1 >=0) {
          return new Feature.PrevPOS(sentence1.get().get(CoreAnnotations.TokensAnnotation.class).get(span1.get().start()-1).tag().equals(sentence2.get().get(CoreAnnotations.TokensAnnotation.class).get(span2.get().start()-1).tag()));
        }
      }
      return null;
    } else if(clazz.equals(Feature.NERTotalMatch.class)) {
      if(getNerForEntityContext(entityOne) != null) {
        //logger.log("Total match fires");
        if(getNerForEntityContext(entityTwo) != null) {
          return new Feature.NERTotalMatch(getNerForEntityContext(entityOne).equals(getNerForEntityContext(entityTwo)));
        }
      }
      return null;
    } else if(clazz.equals(Feature.NameNERMatch.class)) {
        if(ner1.equals(ner2)) {
          countsFeature[7]++;
          return new Feature.NameNERMatch(true);
        } else {
          return new Feature.NameNERMatch(false);
        }
    } else if(clazz.equals(Feature.NERAnyMatch.class)) {
      int count = 0;
      if(span1.isDefined() && span2.isDefined() && sentence1.isDefined() && sentence2.isDefined()) {
        for(int i= span1.get().start(); i < span1.get().end(); i++) {
          for(int j= span2.get().start(); j < span2.get().end(); j++) {
            if(sentence1.get().get(CoreAnnotations.TokensAnnotation.class).get(i).ner().equals(sentence2.get().get(CoreAnnotations.TokensAnnotation.class).get(j).ner())) {
              count++;
              break;
            }
          }
        }
        return new Feature.NERAnyMatch(count);
      }
      return null;
    } else if(clazz.equals(Feature.POSTotalMatch.class)) {
      if(getPOSForEntityContext(entityOne) != null) {
        if(getPOSForEntityContext(entityTwo) != null) {
          return new Feature.POSTotalMatch(getPOSForEntityContext(entityOne).equals(getPOSForEntityContext(entityTwo)));
        }
      }
      return null;
    } else if(clazz.equals(Feature.EntityLengthDiff.class)) {
      double longer = name1.length() > name2.length() ? name1.length() : name2.length();
      countsFeature[6]++;
      return new Feature.EntityLengthDiff(Math.abs(name1.length() - name2.length())/ longer);
    } else if(clazz.equals(Feature.ContextLengthDiff.class)) {
      if(sentence1.isDefined() && sentence2.isDefined()) {
        double longer = sentence1.get().size() > sentence2.get().size() ? sentence1.get().size() : sentence2.get().size();
        return new Feature.ContextLengthDiff(Math.abs(sentence1.get().size() - sentence2.get().size())/ longer);
      }
      return null;
    } else if(clazz.equals(Feature.PrevNER.class)) {
      if(span1.isDefined() && span2.isDefined() && sentence1.isDefined() && sentence2.isDefined()) {
        if(span1.get().start()-1 >= 0 && span2.get().start()-1 >=0) {
          return new Feature.PrevNER(sentence1.get().get(CoreAnnotations.TokensAnnotation.class).get(span1.get().start()-1).ner().equals(sentence2.get().get(CoreAnnotations.TokensAnnotation.class).get(span2.get().start()-1).ner()));
        }
        return null;
      }
      return null;
    } else if(clazz.equals(Feature.NextPOS.class)) {
      if(span1.isDefined() && span2.isDefined() && sentence1.isDefined() && sentence2.isDefined()) {
        if(span1.get().end() < sentence1.get().size() && span2.get().end() < sentence2.get().size()) {
          return new Feature.NextPOS(sentence1.get().get(CoreAnnotations.TokensAnnotation.class).get(span1.get().end()).tag().equals(sentence2.get().get(CoreAnnotations.TokensAnnotation.class).get(span2.get().end()).tag()));
        }
        return null;
      }
      return null;
    } else if(clazz.equals(Feature.NextNER.class)) {
      if(span1.isDefined() && span2.isDefined() && sentence1.isDefined() && sentence2.isDefined()) {
        if(span1.get().end() < sentence1.get().size() && span2.get().end() < sentence2.get().size()) {
          return new Feature.NextNER(sentence1.get().get(CoreAnnotations.TokensAnnotation.class).get(span1.get().end()).ner().equals(sentence2.get().get(CoreAnnotations.TokensAnnotation.class).get(span2.get().end()).ner()));
        }
        return null;
      }
      return null;
    } else if(clazz.equals(Feature.MatchTokens.class)) {
      String[] oneToks, twoToks;
      if(sentence1.isDefined() && sentence2.isDefined()) {
        oneToks = sentence1.get().get(CoreAnnotations.TextAnnotation.class).split("\\s+");
        twoToks = sentence2.get().get(CoreAnnotations.TextAnnotation.class).split("\\s+");
      }
      else {
        oneToks = entityOne.entity.name.split("\\s+");
        twoToks = entityTwo.entity.name.split("\\s+");;
      }
        // Case: acronyms of each other
        //if (AcronymMatcher.isAcronym(higherToks, lowerToks)) { return 1.0; }

        int match = 0;
        // Get number of matching tokens between the two entity sentences
        boolean[] matchedHigherToks = new boolean[oneToks.length];
        boolean[] matchedLowerToks = new boolean[twoToks.length];
        for (int h = 0; h < oneToks.length; ++h) {
          if (matchedHigherToks[h]) { continue; }
          String higherTok = oneToks[h];
          String higherTokNoSpecialChars = Utils.noSpecialChars(higherTok);
          boolean doesMatch = false;
          for (int l = 0; l < twoToks.length; ++l) {
            if (matchedLowerToks[l]) { continue; }
            String lowerTok = twoToks[l];
            String lowerTokNoSpecialCars = Utils.noSpecialChars(lowerTok);
            int minLength = Math.min(lowerTokNoSpecialCars.length(), higherTokNoSpecialChars.length());
            if (higherTokNoSpecialChars.equalsIgnoreCase(lowerTokNoSpecialCars)  // equal
                    ) {
              doesMatch = true;
              matchedHigherToks[h] = true;
              matchedLowerToks[l] = true;
            }
          }
          if (doesMatch) { match += 1; }
        }
        return new Feature.MatchTokens((double) match/ (double) Math.max(oneToks.length, twoToks.length));
    } else if(clazz.equals(Feature.MatchNounTokens.class)) {
      String[] oneToks, twoToks;
      if(sentence1.isDefined() && sentence2.isDefined()) {
        //oneToks = sentence1.get().get(CoreAnnotations.TextAnnotation.class).split("\\s+");
        //twoToks = sentence2.get().get(CoreAnnotations.TextAnnotation.class).split("\\s+");
        countsFeature[14]++;
        // get Corelabels and find tokens

        List<CoreLabel> oneLabels = sentence1.get().get(CoreAnnotations.TokensAnnotation.class);
        List<CoreLabel> twoLabels = sentence2.get().get(CoreAnnotations.TokensAnnotation.class);

        int match = 0;
        // Get number of matching tokens between the two entity sentences
        boolean[] matchedHigherToks = new boolean[oneLabels.size()];
        boolean[] matchedLowerToks = new boolean[twoLabels.size()];
        int count1 = 0;
        int count2 = 0;
        for (int h = 0; h < oneLabels.size(); ++h) {
          if (matchedHigherToks[h]) { continue; }
          String higherTok = oneLabels.get(h).word();
          String higherTokNoSpecialChars = Utils.noSpecialChars(higherTok);
          String higherPos = oneLabels.get(h).tag();
          if(higherPos.startsWith("N")) {
            count1++;
          }
          boolean doesMatch = false;
          for (int l = 0; l < twoLabels.size(); ++l) {
            if (matchedLowerToks[l]) { continue; }
            String lowerTok = twoLabels.get(l).word();
            String lowerTokNoSpecialCars = Utils.noSpecialChars(lowerTok);
            String lowerPos = twoLabels.get(l).tag();
            if(lowerPos.startsWith("N") && h==0) {
              count2++;
            }
            if (higherTokNoSpecialChars.equalsIgnoreCase(lowerTokNoSpecialCars) && (higherPos.startsWith("N") && lowerPos.startsWith("N"))) {
              doesMatch = true;
              matchedHigherToks[h] = true;
              matchedLowerToks[l] = true;
            }
          }
          if (doesMatch) { match += 1; }
        }
        return new Feature.MatchNounTokens(match);
      }
      else {
        return null;
      }

    } else if(clazz.equals(Feature.MatchVerbTokens.class)) {
      String[] oneToks, twoToks;
      if(sentence1.isDefined() && sentence2.isDefined()) {
        //oneToks = sentence1.get().get(CoreAnnotations.TextAnnotation.class).split("\\s+");
        //twoToks = sentence2.get().get(CoreAnnotations.TextAnnotation.class).split("\\s+");
        countsFeature[15]++;
        // get Corelabels and find tokens

        List<CoreLabel> oneLabels = sentence1.get().get(CoreAnnotations.TokensAnnotation.class);
        List<CoreLabel> twoLabels = sentence2.get().get(CoreAnnotations.TokensAnnotation.class);

        int match = 0;
        // Get number of matching tokens between the two entity sentences
        boolean[] matchedHigherToks = new boolean[oneLabels.size()];
        boolean[] matchedLowerToks = new boolean[twoLabels.size()];
        int count1 = 0;
        int count2 = 0;
        for (int h = 0; h < oneLabels.size(); ++h) {
          if (matchedHigherToks[h]) { continue; }
          String higherTok = oneLabels.get(h).word();
          String higherTokNoSpecialChars = Utils.noSpecialChars(higherTok);
          String higherPos = oneLabels.get(h).tag();
          if(higherPos.startsWith("V")) {
            count1++;
          }
          boolean doesMatch = false;
          for (int l = 0; l < twoLabels.size(); ++l) {
            if (matchedLowerToks[l]) { continue; }
            String lowerTok = twoLabels.get(l).word();
            String lowerTokNoSpecialCars = Utils.noSpecialChars(lowerTok);
            String lowerPos = twoLabels.get(l).tag();
            if(lowerPos.startsWith("V") && h==0) {
              count2++;
            }
            if (higherTokNoSpecialChars.equalsIgnoreCase(lowerTokNoSpecialCars) && (higherPos.startsWith("V") && lowerPos.startsWith("V"))) {
              doesMatch = true;
              matchedHigherToks[h] = true;
              matchedLowerToks[l] = true;
            }
          }
          if (doesMatch) { match += 1; }
        }
        return new Feature.MatchVerbTokens(match);
      }
      else {
        return null;
      }

    } else if(clazz.equals(Feature.SentenceAcronym.class)) {
      if(sentence1.isDefined() && sentence2.isDefined()) {
        String[] oneToks = sentence1.get().get(CoreAnnotations.TextAnnotation.class).split("\\W");
        String[] twoToks = sentence2.get().get(CoreAnnotations.TextAnnotation.class).split("\\W");
        return new Feature.SentenceAcronym(AcronymMatcher.isAcronym(oneToks, twoToks));
      }
      return null;
    } else if(clazz.equals(Feature.NameAcronym.class)) {
      String[] oneToks = stripDeterminers(name1).split("\\W");
      String[] twoToks = stripDeterminers(name2).split("\\W");
      if(AcronymMatcher.isAcronym(oneToks, twoToks)) {
        countsFeature[8]++;
        return new Feature.NameAcronym(true);
      } else {
        oneToks = stripDeterminers(name1).split("\\s+");
        twoToks = stripDeterminers(name2).split("\\s+");
        if(Utils.levenshteinDistance(oneToks, twoToks) == 1) {
          if(oneToks.length == twoToks.length) {
            Iterator<String> iter1 = Arrays.asList(oneToks).iterator();
            Iterator<String> iter2 = Arrays.asList(twoToks).iterator();
            String str1 = popNextOrNull(iter1);
            String str2 = popNextOrNull(iter2);
            // find and match acronyms for different token
            while(str1 != null && str2 != null) {
              if(!str1.equals(str2)) {
                if((abbreviations.containsKey(str1.toLowerCase()) && abbreviations.get(str1.toLowerCase()).equalsIgnoreCase(str2)) || (abbreviations.containsKey(str2.toLowerCase()) && abbreviations.get(str2.toLowerCase()).equalsIgnoreCase(str1))) {
                  countsFeature[8]++;
                  return new Feature.NameAcronym(true);
                }
              }
              str1 = popNextOrNull(iter1);
              str2 = popNextOrNull(iter2);
            }
          } else {
            return new Feature.NameAcronym(false);
          }
        }
        return new Feature.NameAcronym(false);
      }
    }
    else {
      throw new IllegalArgumentException("Unregistered feature: " + clazz);
    }
  }

  public Counter<Feature> featurize(Pair<EntityContext, EntityContext> input) {


    Counter<Feature> features = new ClassicCounter<>();

    for(Object o : FEATURES){
      if(o instanceof ConditionalFeature.Specification){
        //(case: singleton feature)
        //Option<Double> count = new Option<Double>(1.0);
        Feature feat = featurize2((ConditionalFeature.Specification) o, input);

        if(feat!=null) {
          Option<Double> count = new Option<Double>(feat.getCount());
          if(count.get() > 0.0){
            features.incrementCount(feat, count.get());
          }
        }
      } else if(o instanceof Pair){
        //(case: pair of features)
        Pair<Class,Class> pair = (Pair<Class,Class>) o;

        Feature featA = feature(pair.first, input);
        Feature featB = feature(pair.second, input);
        if(featA != null && featB != null) {
          Option<Double> countA = new Option<Double>(featA.getCount());
          Option<Double> countB = new Option<Double>(featB.getCount());
          if(countA.get() * countB.get() > 0.0){
            features.incrementCount(new Feature.PairFeature(featA, featB), countA.get() * countB.get());
          }
        }
      }
    }
    features.incrementCount(new Feature.Bias(true), 1.0);
    return features;
  }
  /**
   * Featurize two entity contexts into a datum, which can be passed into a classifier to determine if they
   * refer to the same entity.
   *
   *
   * @param sameEntity True if these do in fact refer to the same entity. This option is irrelevant at test time.
   * @return A datum corresponding to the featurized form of the entity context pair.
   */
  public RVFDatum<Boolean, Feature> featurize(Pair<EntityContext, EntityContext> input, boolean sameEntity) {
    return new RVFDatum<>(featurize(input), sameEntity);
  }

  /**
   * Read nicknames from file
   */
  public static Map<String, String> readNicknames(String classpathOrFile) {
    try {
      Map<String, String> names = new HashMap<String, String>();
      BufferedReader reader = IOUtils.getBufferedReaderFromClasspathOrFileSystem(classpathOrFile);
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
      return new HashMap<String, String>();
    }
  }

  /**
   * Read abbrevations from file
   */
  private Map<String, String> readAbbreviations(String classpathOrFile) {
    try {
      Map<String, String> abbrvs = new HashMap<String, String>();
      BufferedReader reader = IOUtils.getBufferedReaderFromClasspathOrFileSystem(classpathOrFile);
      String line;
      while ((line = reader.readLine()) != null) {
        String[] parts = line.split("\\t");
        String word = parts[1];
        if(!abbrvs.containsKey(word)) {
          abbrvs.put(word.toLowerCase(), parts[0].toLowerCase());
        }
      }
      return abbrvs;
    } catch (IOException e) {
      Redwood.Util.err(e);
      return new HashMap<String, String>();
    }
  }

 /*
 * Get ner tag from Entity Context
 * @param EntityContext
 * @return String : ner tag
  */
  public static String getNerForEntityContext(EntityContext entityOne) {
    if(entityOne.entityTokenSpan.isDefined() && entityOne.sentence.isDefined()) {
      String prevNer = entityOne.sentence.get().get(CoreAnnotations.TokensAnnotation.class).get(entityOne.entityTokenSpan.get().start()).ner();

      for(int i= entityOne.entityTokenSpan.get().start()+1; i < entityOne.entityTokenSpan.get().end(); i++) {
        String currNer = entityOne.sentence.get().get(CoreAnnotations.TokensAnnotation.class).get(i).ner();
        if(prevNer.equals(Props.NER_BLANK_STRING)) {
          if(!currNer.equals(Props.NER_BLANK_STRING)) {
            prevNer = currNer;
          } else {
            continue;
          }
        }
        if( !currNer.equals(Props.NER_BLANK_STRING)  && currNer.equals(prevNer)) {
          prevNer = currNer;
        }
        else if (currNer.equals(Props.NER_BLANK_STRING)) {
        }
        else {
          return null;
        }
      }
      return Props.NER_BLANK_STRING.equals(prevNer) ? null : prevNer;
    }
    else {
      return entityOne.entity.type.name;
    }
  }

  public static String getPOSForEntityContext(EntityContext entityOne) {
    if(entityOne.entityTokenSpan.isDefined() && entityOne.sentence.isDefined()) {
      String prevPos = entityOne.sentence.get().get(CoreAnnotations.TokensAnnotation.class).get(entityOne.entityTokenSpan.get().start()).tag();
      for(int i= entityOne.entityTokenSpan.get().start()+1; i < entityOne.entityTokenSpan.get().end(); i++) {
        String currPos = entityOne.sentence.get().get(CoreAnnotations.TokensAnnotation.class).get(i).tag();
        if(currPos.equals(prevPos)) {
          prevPos = currPos;
        }
        else {
          return null;
        }
      }
      return prevPos;
    }
    return null;
  }

}
