package edu.stanford.nlp.kbp.common;

import java.util.List;

import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.ArrayCoreMap;
import edu.stanford.nlp.util.Pair;
import org.junit.Test;

import static junit.framework.Assert.*;

public class StringFinderTest {
  private boolean verbose = false;
  
  // convenience method to turn a string into a CoreMap
  public CoreMap buildSentence(String sentence) {
    CoreMap result = new ArrayCoreMap();
    result.set(TokensAnnotation.class, CoreMapUtils.tokenize(sentence));
    return result;
  }

  @Test
  public void testExactMatch() {
    StringFinder finder = new StringFinder("exact match");
    if(verbose) System.out.println("Using finder: " + finder);
    
    CoreMap testSentence = buildSentence("exact match");
    assertTrue(finder.matches(testSentence));
  }

  @Test
  public void testSimple() {
    StringFinder finder = new StringFinder("simple test");
    if(verbose) System.out.println("Using finder: " + finder);

    CoreMap testSentence = buildSentence("This is a simple test");
    assertTrue(finder.matches(testSentence));

    CoreMap testMiss = buildSentence("This should not match");
    assertFalse(finder.matches(testMiss));
  }

  @Test
  public void testComma() {
    StringFinder finder = 
      new StringFinder("University of California Berkeley");
    if(verbose) System.out.println("Using finder: " + finder);

    CoreMap testSentence = 
      buildSentence("foo University of California, Berkeley bar");
    assertTrue(finder.matches(testSentence));    

    testSentence = buildSentence("foo University of California Berkeley bar");
    assertTrue(finder.matches(testSentence));

    testSentence = buildSentence("foo University of Californiaz Berkeley bar");
    assertFalse(finder.matches(testSentence));

    finder = new StringFinder("University of California, Berkeley");
    if(verbose) System.out.println("Using finder: " + finder);
    testSentence = buildSentence("foo University of California, Berkeley bar");
    assertTrue(finder.matches(testSentence));    

    testSentence = buildSentence("foo University of California Berkeley bar");
    assertTrue(finder.matches(testSentence));

    testSentence = buildSentence("foo University of Californiaz Berkeley bar");
    assertFalse(finder.matches(testSentence));
  }

  @Test
  public void testPeriod() {
    StringFinder finder = new StringFinder("Pentax Corp.");
    if(verbose) System.out.println("Using finder: " + finder);
    
    CoreMap testSentence = buildSentence("Blah pentax corp. blah");
    assertTrue(finder.matches(testSentence));    

    testSentence = buildSentence("Blah pentax corp blah");
    assertTrue(finder.matches(testSentence));    
 
    finder = new StringFinder("Pentax Corp");
    if(verbose) System.out.println("Using finder: " + finder);
    
    testSentence = buildSentence("Blah pentax corp. blah");
    assertTrue(finder.matches(testSentence));    

    testSentence = buildSentence("Blah pentax corp blah");
    assertTrue(finder.matches(testSentence));    
  }

  @Test
  public void testMultipleStrings() {
    StringFinder finder = new StringFinder("foo", "bar");
    if(verbose) System.out.println("Using finder: " + finder);

    CoreMap testSentence = buildSentence("blah foo blah");
    assertTrue(finder.matches(testSentence));
    testSentence = buildSentence("blah bar blah");
    assertTrue(finder.matches(testSentence));
    testSentence = buildSentence("blah baz blah");
    assertFalse(finder.matches(testSentence));
  }

  @Test
  public void testDash() {
    StringFinder finder = 
      new StringFinder("Nottingham-Spirk Design Associates");
    if(verbose) System.out.println("Using finder: " + finder);

    CoreMap testSentence = 
      buildSentence("foo Nottingham-Spirk Design Associates bar");
    assertTrue(finder.matches(testSentence));
    testSentence = buildSentence("foo Nottingham Spirk Design Associates bar");
    assertTrue(finder.matches(testSentence));
    testSentence = buildSentence("foo NottinghamSpirk Design Associates bar");
    assertTrue(finder.matches(testSentence));
  }

  @Test
  public void testPlus() {
    StringFinder finder = new StringFinder("+44");
    if(verbose) System.out.println("Using finder: " + finder);
    CoreMap testSentence = buildSentence("foo +44 blah");
    assertTrue(finder.matches(testSentence));
    testSentence = buildSentence("foo 44 blah");
    assertFalse(finder.matches(testSentence));
  }

  @Test
  public void testParens() {
    StringFinder finder = new StringFinder("Pre(Thing");
    if(verbose) System.out.println("Using finder: " + finder);
    CoreMap testSentence = buildSentence("foo Pre(Thing blah");
    assertTrue(finder.matches(testSentence));
    testSentence = buildSentence("foo PreThing blah");
    assertFalse(finder.matches(testSentence));
  }

  @Test
  public void testWhereItMatches() {
    StringFinder finder = new StringFinder("London, UK");
    if(verbose) System.out.println("Using finder: " + finder);
    CoreMap testSentence = buildSentence("The London, UK Symphony Orchestra plays in London UK");
    List<Pair<Integer, Integer>> matches = finder.whereItMatches(testSentence);
    if(verbose) System.out.println("Matches: " + matches);
    assertTrue(matches.size() == 2);
    assertTrue(matches.get(0).first() == 4 && matches.get(0).second() == 13);
    assertTrue(matches.get(1).first() == 42 && matches.get(1).second() == 51);
  }
}
