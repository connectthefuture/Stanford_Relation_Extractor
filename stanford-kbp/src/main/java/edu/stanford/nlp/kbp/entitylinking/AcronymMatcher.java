package edu.stanford.nlp.kbp.entitylinking;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.StringUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;

/**
 * TODO(gabor) JavaDoc
 *
 * @author Gabor Angeli
 */
public class AcronymMatcher {
  private static final Pattern discardPattern = Pattern.compile("[-._]");

  private static List<String> getTokenStrs(List<CoreLabel> tokens)
  {
    List<String> mainTokenStrs = new ArrayList<String>(tokens.size());
    for (CoreLabel token:tokens) {
      String text = token.get(CoreAnnotations.TextAnnotation.class);
      mainTokenStrs.add(text);
    }
    return mainTokenStrs;
  }

  private static List<String> getMainTokenStrs(List<CoreLabel> tokens)
  {
    List<String> mainTokenStrs = new ArrayList<String>(tokens.size());
    for (CoreLabel token:tokens) {
      String text = token.get(CoreAnnotations.TextAnnotation.class);
      if (!text.isEmpty() && ( text.length() >= 4 || Character.isUpperCase(text.charAt(0))) ) {
        mainTokenStrs.add(text);
      }
    }
    return mainTokenStrs;
  }

  private static List<String> getMainTokenStrs(String[] tokens)
  {
    List<String> mainTokenStrs = new ArrayList<String>(tokens.length);
    for (String text:tokens) {
      if ( !text.isEmpty() && ( text.length() >= 4 || Character.isUpperCase(text.charAt(0)) ) ) {
        mainTokenStrs.add(text);
      }
    }
    return mainTokenStrs;
  }

  public static List<String> getMainStrs(List<String> tokens)
  {
    List<String> mainTokenStrs = new ArrayList<String>(tokens.size());
    for (String text:tokens) {
      if ( !text.isEmpty() && (text.length() >= 4 || Character.isUpperCase(text.charAt(0))) ) {
        mainTokenStrs.add(text);
      }
    }
    return mainTokenStrs;
  }

  public static boolean isAcronym(String str, String[] tokens) {
    return isAcronymImpl(str, Arrays.asList(tokens));
  }

  // Public static utility methods
  public static boolean isAcronymImpl(String str, List<String> tokens)
  {
    str = discardPattern.matcher(str).replaceAll("");
    if (str.length() == tokens.size()) {
      for (int i = 0; i < str.length(); i++) {
        char ch = Character.toUpperCase(str.charAt(i));
        if ( !tokens.get(i).isEmpty() &&
            Character.toUpperCase(tokens.get(i).charAt(0)) != ch ) {
          return false;
        }
      }
      return true;
    } else {
      return false;
    }
  }

  public static boolean isAcronym(String str, List<?> tokens)
  {
    List<String> strs = new ArrayList<String>(tokens.size());
    for (Object tok : tokens) { strs.add(tok instanceof String ? tok.toString() : ((CoreLabel) tok).word()); }
    return isAcronymImpl(str, strs);
  }

  /**
   * Returns true if either chunk1 or chunk2 is acronym of the other
   * @return true if either chunk1 or chunk2 is acronym of the other
   */
  public static boolean isAcronym(CoreMap chunk1, CoreMap chunk2)
  {
    String text1 = chunk1.get(CoreAnnotations.TextAnnotation.class);
    String text2 = chunk2.get(CoreAnnotations.TextAnnotation.class);
    if (text1.length() <= 1 || text2.length() <= 1) { return false; }
    List<String> tokenStrs1 = getTokenStrs(chunk1.get(CoreAnnotations.TokensAnnotation.class));
    List<String> tokenStrs2 = getTokenStrs(chunk2.get(CoreAnnotations.TokensAnnotation.class));
    boolean isAcro = isAcronymImpl(text1, tokenStrs2) || isAcronymImpl(text2, tokenStrs1);
    if (!isAcro) {
      tokenStrs1 = getMainTokenStrs(chunk1.get(CoreAnnotations.TokensAnnotation.class));
      tokenStrs2 = getMainTokenStrs(chunk2.get(CoreAnnotations.TokensAnnotation.class));
      isAcro = isAcronymImpl(text1, tokenStrs2) || isAcronymImpl(text2, tokenStrs1);
    }
    return isAcro;
  }

  /** @see edu.stanford.nlp.kbp.entitylinking.AcronymMatcher#isAcronym(edu.stanford.nlp.util.CoreMap, edu.stanford.nlp.util.CoreMap) */
  public static boolean isAcronym(String[] chunk1, String[] chunk2)
  {
    String text1 = StringUtils.join(chunk1);
    String text2 = StringUtils.join(chunk2);
    if (text1.length() <= 1 || text2.length() <= 1) { return false; }
    List<String> tokenStrs1 = Arrays.asList(chunk1);
    List<String> tokenStrs2 = Arrays.asList(chunk2);
    boolean isAcro = isAcronymImpl(text1, tokenStrs2) || isAcronymImpl(text2, tokenStrs1);
    if (!isAcro) {
      tokenStrs1 = getMainTokenStrs(chunk1);
      tokenStrs2 = getMainTokenStrs(chunk2);
      isAcro = isAcronymImpl(text1, tokenStrs2) || isAcronymImpl(text2, tokenStrs1);
    }
    return isAcro;
  }


  /** @see edu.stanford.nlp.kbp.entitylinking.AcronymMatcher#isAcronym(edu.stanford.nlp.util.CoreMap, edu.stanford.nlp.util.CoreMap) */
  public static boolean isPartialAcronym(String[] chunk1, String[] chunk2)
  {
    String text1 = StringUtils.join(chunk1);
    String text2 = StringUtils.join(chunk2);
    if (text1.length() <= 1 || text2.length() <= 1) { return false; }
    List<String> tokenStrs1 = Arrays.asList(chunk1);
    List<String> tokenStrs2 = Arrays.asList(chunk2);
    boolean isAcro = isPartialAcronymImpl(text1, tokenStrs2) || isPartialAcronymImpl(text2, tokenStrs1);
    if (!isAcro) {
      tokenStrs1 = getMainTokenStrs(chunk1);
      tokenStrs2 = getMainTokenStrs(chunk2);
      isAcro = isAcronymImpl(text1, tokenStrs2) || isAcronymImpl(text2, tokenStrs1);
    }
    return isAcro;
  }

  // Public static utility methods
  public static boolean isPartialAcronymImpl(String str, List<String> tokens)
  {
    str = discardPattern.matcher(str).replaceAll("");
    if (str.length() == tokens.size()) {
      for (int i = 0; i < str.length(); i++) {
        char ch = Character.toUpperCase(str.charAt(i));
        if ( !tokens.get(i).isEmpty() &&
                Character.toUpperCase(tokens.get(i).charAt(0)) != ch ) {
          return false;
        }
      }
      return true;
    } else {
      return false;
    }
  }

  public static boolean isFancyAcronym(String[] chunk1, String[] chunk2) {
    String text1 = StringUtils.join(chunk1);
    String text2 = StringUtils.join(chunk2);
    if (text1.length() <= 1 || text2.length() <= 1) { return false; }
    List<String> tokenStrs1 = Arrays.asList(chunk1);
    List<String> tokenStrs2 = Arrays.asList(chunk2);
    return isFancyAcronymImpl(text1, tokenStrs2) || isFancyAcronymImpl(text2, tokenStrs1);
  }

  public static boolean isFancyAcronymImpl(String str, List<String> tokens) {
    str = discardPattern.matcher(str).replaceAll("");
    String text = StringUtils.join(tokens);
    int prev_index = 0;
    for(int i=0; i < str.length(); i++) {
      char ch = str.charAt(i);
      if(text.indexOf(ch) != -1) {
        prev_index = text.indexOf(ch, prev_index);
        if(prev_index == -1) {
          return false;
        }
      }
      else {
        return false;
      }
    }
    return true;
  }
}
