package edu.stanford.nlp.kbp.slotfilling.process;

import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.kbp.common.KBPSlotFill;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.semgraph.SemanticGraph;

import java.io.Serializable;
import java.util.*;

/**
 * An encapsulation of the relevant information for extracting features for relation classification.
 * In many ways, this is more convenient than the raw sentence, as it allows for easier unit testing, and in
 * general is more transparent than annotations stuck onto sentences.
 *
 * @author Gabor Angeli
 */
@SuppressWarnings("unchecked")
public class Featurizable implements Serializable {
  private static final long serialVersionUID = 1l;

  public final Span subj;
  public final Span obj;
  public final List<CoreLabel> tokens;
  public final SemanticGraph dependencies;
  public final Collection<KBPSlotFill> openIE;

  public Featurizable(Span subj, Span obj, List<CoreLabel> tokens, SemanticGraph dependencies,
                      Collection<KBPSlotFill> openIE) {
    // Set input
    this.subj = subj;
    this.obj = obj;
    this.tokens = tokens;
    this.dependencies = dependencies;
    this.openIE = openIE;
    // Check input
    assert subj != null;
    assert obj != null;
    assert subj.start() < subj.end();
    assert obj.start() < obj.end();
    assert tokens != null;
    assert tokens.size() >= subj.end();
    assert tokens.size() >= obj.end();
    assert dependencies != null;
  }
}
