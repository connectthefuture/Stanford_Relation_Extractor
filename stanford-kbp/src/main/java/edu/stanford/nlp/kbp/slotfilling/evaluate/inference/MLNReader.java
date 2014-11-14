package edu.stanford.nlp.kbp.slotfilling.evaluate.inference;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.StringReader;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Simple parser for MLNs
 */
public class MLNReader {

  public static MLNText parse(BufferedReader in) throws IOException {
    Pattern PREDICATE_REGEXP = Pattern.compile("([^-0-9][^\\(]*)\\(\\s*([^,]+)\\s*,\\s*([^\\)]+)\\s*\\)");
    Pattern CLAUSE_REGEXP = Pattern.compile("!?([^\\(]+)\\(\\s*([^,]+)\\s*,\\s*([^\\)]+)\\s*\\)");
    Pattern EQUALS_CLAUSE_REGEXP = Pattern.compile("\\s*([^,]+)\\s*(!?=)\\s*([^,.]+)");

    MLNText text = new MLNText();

    String line;
    while ( (line = in.readLine()) != null ) {
      // Get rid of comments.
      line = Pattern.compile("//.*$").matcher(line).replaceAll("");
      line = line.trim();
      if( line.length() == 0 || line.startsWith("//") ) continue;
      else if (PREDICATE_REGEXP.matcher(line).matches()) {
        Matcher result = PREDICATE_REGEXP.matcher(line);
        if (!result.matches()) { throw new AssertionError(); }
        result.start();
        MLNText.Predicate pred = new MLNText.Predicate(result.group(1), result.group(2), result.group(3));
        text.predicates.add(pred);
      } else {
        // Try to match a deterministic rule.
        double weight;
        if( line.endsWith(".") ) {
          weight = Double.POSITIVE_INFINITY;
          line = line.substring(0,line.length()-1);
        } else {
          String[] elems = line.split(" ", 2);
          weight = Double.parseDouble(elems[0]);
          line = elems[1];
        }

        MLNText.Rule rule = new MLNText.Rule();
        rule.weight = weight;

        // HACK! This is to prevent relations_with_v_in_them from being horribly split.
        for( String clause : line.replaceAll("\\)v", ")  v  ").split("\\s+v\\s+") ) {
          clause = clause.trim();
          if( CLAUSE_REGEXP.matcher(clause).matches() ) {
            Matcher matcher = CLAUSE_REGEXP.matcher(clause);
            if (!matcher.matches()) { throw new AssertionError(); }
            MLNText.Literal literal = new MLNText.Literal(
                    !clause.startsWith("!"),
                    matcher.group(1),
                    matcher.group(2), matcher.group(3));
            rule.literals.add(literal);
          } else if( EQUALS_CLAUSE_REGEXP.matcher(clause).matches() ) {
            Matcher matcher = EQUALS_CLAUSE_REGEXP.matcher(clause);
            if (!matcher.matches()) { throw new AssertionError(); }
            MLNText.Literal literal = new MLNText.Literal(
                    !matcher.group(2).startsWith("!"),
                    "=",
                    matcher.group(1), matcher.group(3));
            rule.literals.add(literal);
          } else {
            System.err.println(clause);
            throw new IllegalArgumentException("Invalid line of rule file (didn't match regexp): " + line);
          }
        }
        text.rules.add(rule);
      }
    }
    return text;
  }

  public static MLNText parse(String in) {
    try {
      return parse(new BufferedReader(new StringReader(in)));
    } catch (IOException ex) {
      throw new RuntimeException(ex);
    }
  }

}
