package edu.stanford.nlp.kbp.common;

import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.util.CoreMap;

/**
* TODO(gabor) JavaDoc
*
* @author Gabor Angeli
*/
public class Mention {
  public final CoreMap sentence;
  public final Span spanInSentence;
  public final KBPEntity entity;

  public Mention(CoreMap sentence, Span spanInSentence, KBPEntity entity) {
    this.sentence = sentence;
    this.spanInSentence = spanInSentence;
    this.entity = entity;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (!(o instanceof Mention)) return false;
    Mention mention = (Mention) o;
    return entity.equals(mention.entity) && sentence == mention.sentence && spanInSentence.equals(mention.spanInSentence);
  }

  @Override
  public int hashCode() {
    int result = sentence.hashCode();
    result = 31 * result + spanInSentence.hashCode();
    result = 31 * result + entity.hashCode();
    return result;
  }

  @Override
  public String toString() {
    return entity + "@" + spanInSentence;
  }
}
