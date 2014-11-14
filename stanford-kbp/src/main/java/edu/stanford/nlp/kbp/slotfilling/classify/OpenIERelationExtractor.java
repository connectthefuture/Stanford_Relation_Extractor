package edu.stanford.nlp.kbp.slotfilling.classify;

import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.pipeline.Annotation;
import java.util.List;

/**
 * An interface to Keenon's Exemplar augmenting the official relations with OpenIE extractions.
 *
 * @author Gabor Angeli
 */
public class OpenIERelationExtractor {
  public List<KBPSlotFill> extractRelations(Annotation doc) {
    throw new IllegalStateException("Exemplar has been deprecated to make releasing KBP easier.");
  }
}
