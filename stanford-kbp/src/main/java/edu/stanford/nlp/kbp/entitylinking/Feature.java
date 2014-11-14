package edu.stanford.nlp.kbp.entitylinking;

import edu.stanford.nlp.util.Pair;

import java.io.Serializable;
import java.util.Set;

/**
 * Created by melvin on 3/20/14.
 */
  /**
   * @author Gabor Angeli (angeli at cs.stanford)
   */
  public interface Feature extends Serializable {

    //-----------------------------------------------------------
    // TEMPLATE FEATURE TEMPLATES
    //-----------------------------------------------------------
    public double getCount();

    public static class PairFeature implements Feature {
      public final Pair<Feature,Feature> content;
      public PairFeature(Feature a, Feature b){ this.content = Pair.makePair(a, b); }
      public String toString(){ return content.toString(); }
      public boolean equals(Object o){ return o instanceof PairFeature && ((PairFeature) o).content.equals(content); }
      public int hashCode(){ return content.hashCode(); }
      public double getCount(){return 1.0;}
    }

    public static abstract class Indicator implements Feature {
      public final boolean value;
      public Indicator(boolean value){ this.value = value; }
      public boolean equals(Object o){ return o instanceof Indicator && o.getClass().equals(this.getClass()) && ((Indicator) o).value == value; }
      public int hashCode(){
        return this.getClass().hashCode() ^ Boolean.valueOf(value).hashCode(); }
      public String toString(){
        return this.getClass().getSimpleName() + "(" + value + ")"; }
      public double getCount(){return 1.0;}
    }

    public static abstract class IntIndicator implements Feature {
      public final int value;
      public IntIndicator(int value){ this.value = value; }
      public boolean equals(Object o){ return o instanceof IntIndicator && o.getClass().equals(this.getClass()) && ((IntIndicator) o).value == value; }
      public int hashCode(){
        return this.getClass().hashCode() ^ value;
      }
      public String toString(){ return this.getClass().getSimpleName() + "(" + value + ")"; }
      public double getCount(){return 1.0;}
    }

    public static abstract class RealValuedFeature implements Feature {
      public double value;
      public double count;
      public RealValuedFeature(double value){ this.count = value; }
      public boolean equals(Object o){ return o instanceof RealValuedFeature; }
      public int hashCode(){
        return this.getClass().hashCode();
      }
      public String toString(){ return this.getClass().getSimpleName() + "()"; }
      public double getCount(){return this.count;}
    }

    public static abstract class BucketIndicator implements Feature {
      public final int bucket;
      public final int numBuckets;
      public BucketIndicator(int value, int max, int numBuckets){
        this.numBuckets = numBuckets;
        bucket = value * numBuckets / max;
        if(bucket < 0 || bucket >= numBuckets){ throw new IllegalStateException("Bucket out of range: " + value + " max="+max+" numbuckets="+numBuckets); }
      }
      public boolean equals(Object o){ return o instanceof BucketIndicator && o.getClass().equals(this.getClass()) && ((BucketIndicator) o).bucket == bucket; }
      public int hashCode(){ return this.getClass().hashCode() ^ bucket; }
      public String toString(){ return this.getClass().getSimpleName() + "(" + bucket + "/" + numBuckets + ")"; }
      public double getCount(){return 1.0;}
    }

    public static abstract class Placeholder implements Feature {
      public Placeholder(){ }
      public boolean equals(Object o){ return o instanceof Placeholder && o.getClass().equals(this.getClass()); }
      public int hashCode(){ return this.getClass().hashCode(); }
      public String toString(){ return this.getClass().getSimpleName(); }
      public double getCount(){return 1.0;}
    }

    public static abstract class StringIndicator implements Feature {
      public final String str;
      public StringIndicator(String str){ this.str = str; }
      public boolean equals(Object o){ return o instanceof StringIndicator && o.getClass().equals(this.getClass()) && ((StringIndicator) o).str.equals(this.str); }
      public int hashCode(){ return this.getClass().hashCode() ^ str.hashCode(); }
      public String toString(){ return this.getClass().getSimpleName() + "(" + str + ")"; }
      public double getCount(){return 1.0;}
    }

    public static abstract class SetIndicator implements Feature {
      public final Set<String> set;
      public SetIndicator(Set<String> set){ this.set = set; }
      public boolean equals(Object o){ return o instanceof SetIndicator && o.getClass().equals(this.getClass()) && ((SetIndicator) o).set.equals(this.set); }
      public int hashCode(){ return this.getClass().hashCode() ^ set.hashCode(); }
      public String toString(){
        StringBuilder b = new StringBuilder();
        b.append(this.getClass().getSimpleName());
        b.append("( ");
        for(String s : set){
          b.append(s).append(" ");
        }
        b.append(")");
        return b.toString();
      }
      public double getCount(){return 1.0;}
    }

  /*
   * TODO: If necessary, add new feature types
   */

    //-----------------------------------------------------------
    // REAL FEATURE TEMPLATES
    //-----------------------------------------------------------



    public static class ExactMatch extends Indicator {
      public ExactMatch(boolean exactMatch){ super(exactMatch); }
    }

    public static class HeadMatch extends Indicator {
      public HeadMatch(boolean headMatch){ super(headMatch); }
    }

    public static class NickNameMatch extends Indicator {
      public NickNameMatch(boolean nickNameMatch){ super(nickNameMatch); }
    }

    public static class LastNameMatch extends Indicator {
      public LastNameMatch(boolean lastNameMatch){ super(lastNameMatch); }
    }

    public static class FirstNameMatch extends Indicator {
      public FirstNameMatch(boolean firstNameMatch){ super(firstNameMatch); }
    }

    public static class FuzzyNameMatch extends Indicator {
      public FuzzyNameMatch(boolean fuzzyNameMatch){ super(fuzzyNameMatch); }
    }

    public static class EditDistance extends IntIndicator {
      public EditDistance(int editDistance){ super(editDistance); }
    }

    public static class TokenEditDistance extends IntIndicator {
      public TokenEditDistance(int tokenEditDistance){ super(tokenEditDistance); }
    }

    public static class FirstNameEditDistance extends RealValuedFeature {
      public FirstNameEditDistance(double firstNameEditDistance){ super(firstNameEditDistance); }
    }

    public static class MiddleNameEditDistance extends RealValuedFeature {
      public MiddleNameEditDistance(double middleNameEditDistance){ super(middleNameEditDistance); }
    }

    public static class NERTotalMatch extends Indicator {
      public NERTotalMatch(boolean nerTotalMatch){ super(nerTotalMatch); }
    }

    public static class POSTotalMatch extends Indicator {
      public POSTotalMatch(boolean posTotalMatch){ super(posTotalMatch); }
    }

    public static class EntityLengthDiff extends RealValuedFeature {
      public EntityLengthDiff(double lengthDiff){ super(lengthDiff); }
    }

    public static class ContextLengthDiff extends RealValuedFeature {
      public ContextLengthDiff(double lengthDiff){ super(lengthDiff); }
    }

    public static class NERAnyMatch extends IntIndicator {
      public NERAnyMatch(int nerAnyMatch){ super(nerAnyMatch); }
    }

    public static class PrevPOS extends Indicator {
      public PrevPOS(boolean prevPOS){ super(prevPOS); }
    }

    public static class PrevNER extends Indicator {
      public PrevNER(boolean prevNER){ super(prevNER); }
    }

    public static class NextPOS extends Indicator {
      public NextPOS(boolean NextPOS){ super(NextPOS); }
    }

    public static class NextNER extends Indicator {
      public NextNER(boolean NextNER){ super(NextNER); }
    }

    public static class NameNERMatch extends Indicator {
      public NameNERMatch(boolean NameNERMatch) { super(NameNERMatch); }
    }

    public static class NamePOSMatch extends Indicator {
      public NamePOSMatch(boolean NamePOSMatch) { super(NamePOSMatch); }
    }

    public static class MatchTokens extends RealValuedFeature {
      public MatchTokens(double matchTokens){ super(matchTokens); }
    }

    public static class MatchNounTokens extends IntIndicator {
      public MatchNounTokens(int matchNounTokens){ super(matchNounTokens); }
    }

    public static class MatchVerbTokens extends IntIndicator {
      public MatchVerbTokens(int matchVerbTokens){ super(matchVerbTokens); }
    }

    public static class SentenceAcronym extends Indicator {
      public SentenceAcronym(boolean acronym){ super(acronym); }
    }

    public static class NameAcronym extends Indicator {
      public NameAcronym(boolean acronym){ super(acronym); }
    }

    public static class DifferentBy extends StringIndicator {
      public DifferentBy(String differentBy) { super(differentBy); }
    }

    public static class Bias extends Indicator {
      public Bias(boolean bias){ super(bias); }
    }

  }


