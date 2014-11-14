package edu.stanford.nlp.kbp.slotfilling.evaluate;

import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.math.ArrayMath;
import java.util.function.Function;
import edu.stanford.nlp.util.IdentityHashSet;
import edu.stanford.nlp.util.Pair;

import java.util.*;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * A class for simple heuristic consistency checks on slot fills, in the same vein
 * as the old system.
 *
 * @author Gabor Angeli
 */
public abstract class HeuristicSlotfillPostProcessor extends SlotfillPostProcessor {

  /**
   * A class to manage modifications to the active slots vector.
   * On creation, it will mutate the passed slotsActive array;
   * when the modification is complete, call restoreAndReturn,
   * which passes on the passed return value, but also restores the
   * state of the slotsActive vector.
   */
  private static class GibbsState {
    private final boolean[] slotsActive;
    private final boolean savedDeactivatedValue;
    private final boolean savedActivatedValue;
    private final boolean savedActivated2Value;
    private final int toDeactivate;
    private final int toActivate;
    private final int toActivate2;

    private boolean isRestored = false;

    public GibbsState(boolean[] slotsActive, int toDeactivate, int toActivate, int toActivate2) {
      this.slotsActive = slotsActive;
      this.toDeactivate = toDeactivate;
      this.toActivate = toActivate;
      this.toActivate2 = toActivate2;
      this.savedDeactivatedValue = slotsActive[toDeactivate];
      this.savedActivatedValue = slotsActive[toActivate];
      this.savedActivated2Value = slotsActive[toActivate2];
      slotsActive[toDeactivate] = false;
      slotsActive[toActivate] = true;
      slotsActive[toActivate2] = true;
    }

    public <E> E restoreAndReturn(E valueToReturn) {
      if (isRestored) {
        throw new IllegalStateException("Using a Gibbs state twice!");
      }
      slotsActive[toDeactivate] = savedDeactivatedValue;
      slotsActive[toActivate] = savedActivatedValue;
      slotsActive[toActivate2] = savedActivated2Value;
      isRestored = true;
      return valueToReturn;
    }
  }

  private boolean blockGibbsCanTransition(KBPEntity pivot, KBPSlotFill[] slotFills, GibbsState state) {
    return state.restoreAndReturn(isConsistent(pivot, slotFills, state.slotsActive));
  }

  private boolean isConsistent(KBPEntity pivot, KBPSlotFill[] slotFills, boolean[] slotsActive) {
    // -- Singleton Consistency
    for (int i = 0; i < slotFills.length; ++i) {
      if (slotsActive[i] && !isValidSlotAndRewrite(pivot, slotFills[i]).isDefined()) { return false; }
    }

    // -- Pairwise Consistency
    for (int i = 0; i < slotFills.length; ++i) {
      for (int j = i + 1; j < slotFills.length; ++j) {
        if (slotsActive[i] && slotsActive[j] &&
            !pairwiseKeepLowerScoringFill(pivot, slotFills[i], slotFills[j])) {
          return false;
        }
      }
    }

    // -- Hold-one-out Consistency
    // (create set)
    IdentityHashSet<KBPSlotFill> others = new IdentityHashSet<>();
    for (int i = 0; i < slotFills.length; ++i) {
      if (slotsActive[i]) { others.add(slotFills[i]); }
    }
    // (check consistency for all active elements)
    for (int i = 0; i < slotFills.length; ++i) {
      if (slotsActive[i]) {
        others.remove(slotFills[i]);
        if (!leaveOneOutKeepHeldOutSlot(pivot, others, slotFills[i])) { return false; }
        others.add(slotFills[i]);
      }
    }

    // -- Everything Passes
    return true;
  }

  private int greedyEnableSlotsInPlace(KBPEntity pivot, KBPSlotFill[] sortedSlots, boolean[] slotsActive) {
    int slotsEnabled = 0;
    for (int i = 0; i < sortedSlots.length; ++i) {
      if (blockGibbsCanTransition(pivot, sortedSlots, new GibbsState(slotsActive, i, i, i))) {
        slotsActive[i] = true;
        slotsEnabled += 1;
      } else {
        assert !slotsActive[i];
      }
    }
    return slotsEnabled;
  }


  private List<KBPSlotFill> filterStep(KBPEntity pivot, List<KBPSlotFill> slotFills, GoldResponseSet checklist) {
    List<KBPSlotFill> filteredSlots = slotFills;

    // -- Singleton Consistency
    List<KBPSlotFill> withBlatantViolationsFiltered = new ArrayList<>();
    for (KBPSlotFill slotFill : filteredSlots) {
      Maybe<KBPSlotFill> maybeRewritten = isValidSlotAndRewrite(pivot, slotFill);
      for (KBPSlotFill rewritten : maybeRewritten) {
        if (!rewritten.equals(slotFill)) {
          checklist.discardRewritten(slotFill);
          checklist.registerResponse(rewritten);
        }
        withBlatantViolationsFiltered.add(Props.TEST_CONSISTENCY_REWRITE ? rewritten : slotFill);
      }
      if (!maybeRewritten.isDefined()) {
        checklist.discardInconsistent(slotFill);
      }
    }
    filteredSlots = withBlatantViolationsFiltered;

    // -- Nonlocal consistency
    KBPSlotFill[] sortedSlots = filteredSlots.toArray(new KBPSlotFill[filteredSlots.size()]);
    Arrays.sort(sortedSlots);
    boolean[] slotsActive = new boolean[sortedSlots.length];
    int slotsEnabled = greedyEnableSlotsInPlace(pivot, sortedSlots, slotsActive);
    // (pass 1: greedy)
    assert (isConsistent(pivot, sortedSlots, slotsActive));
    // (pass 2: pairwise hops)
    if (Props.TEST_CONSISTENCY_GIBBSOBJECTIVE != Props.GibbsObjective.TOP) {
      Function<Pair<boolean[], KBPSlotFill[]>, Double> objectiveFn = getObjective(Props.TEST_CONSISTENCY_GIBBSOBJECTIVE);

      // Sample and greedy hill climb
      //   variables
      boolean[] argmax = new boolean[slotsActive.length];
      System.arraycopy(slotsActive, 0, argmax, 0, slotsActive.length);
      double max = objectiveFn.apply(Pair.makePair(slotsActive, sortedSlots));
      Random rand = new Random(42);
      int[] enableOrder = CollectionUtils.seq(slotsActive.length);
      log("initial objective: " + max);
      //   sample
      for (int i = 0; i < Props.TEST_CONSISTENCY_MIXINGTIME; ++i) {
        Arrays.fill(slotsActive, false);
        ArrayMath.shuffle(enableOrder, rand);
        for (int toEnable : enableOrder) {
          if (blockGibbsCanTransition(pivot, sortedSlots, new GibbsState(slotsActive, toEnable, toEnable, toEnable))) {
            slotsActive[toEnable] = true;
          }
        }
        double newObjective = objectiveFn.apply(Pair.makePair(slotsActive, sortedSlots));
        if (newObjective > max) {
          max = newObjective;
          System.arraycopy(slotsActive, 0, argmax, 0, slotsActive.length);
          log("found higher objective: " + max);
        }
      }
      //   save result
      slotsActive = argmax;
      slotsEnabled = 0;
      for (boolean active : slotsActive) { slotsEnabled += active ? 1 : 0; }

      /*
      // ^^ Alternative to above ^^
      // Gibbs Hill climbing: while( not converged ): disable one and enable two slots.
      boolean converged = false;
      double objective = objectiveFn.apply(Pair.makePair(slotsActive, sortedSlots));
      while (!converged) {
        log("hill climbing...");
        converged = true;
        for (int off = 0; off < slotsActive.length; ++off) {
          for (int on = 0; on < slotsActive.length; ++on) {
            for (int on2 = 0; on2 < slotsActive.length; ++on2) {
              double candidateObjective = new GibbsState(slotsActive, off, on, on2).restoreAndReturn(objectiveFn.apply(Pair.makePair(slotsActive, sortedSlots)));
              if (candidateObjective > objective &&
                  blockGibbsCanTransition(pivot, sortedSlots, new GibbsState(slotsActive, off, on, on2))) {
                log("replacing " + sortedSlots[off] + " with " + sortedSlots[on] + (on == on2 ? "" : " and " + sortedSlots[on2]));
                slotsActive[off] = false;
                slotsActive[on] = true;
                slotsActive[on2] = true;
                converged = false;
                objective = candidateObjective;
              }
            }
          }
        }
      }
      log("converged.");
      */
    }
    assert (isConsistent(pivot, sortedSlots, slotsActive));
    // (copy list)
    List<KBPSlotFill> withGlobalConsistency = new ArrayList<>();
    for (int i = 0; i < sortedSlots.length; ++i) {
      if (slotsActive[i]) {
        withGlobalConsistency.add(sortedSlots[i]);
      } else {
        checklist.discardInconsistent(sortedSlots[i]);
      }
    }
    filteredSlots = withGlobalConsistency;

    // -- Return
    return filteredSlots;

  }

  private Function<Pair<boolean[],KBPSlotFill[]>,Double> getObjective(Props.GibbsObjective type) {
    switch (type) {
      // Fill as many of te top slots as possible
      case TOP:
        return in -> {
          throw new IllegalStateException("No well defined objective for GibbsObjective.TOP");
        };
      // Optimize for the maximum sum score of the slots filled
      case SUM:
        return in -> {
          boolean[] mask = in.first;
          KBPSlotFill[] fills = in.second;
          double sum = 0.0;
          for (int i = 0; i < mask.length; ++i) {
            if (mask[i]) { sum += fills[i].score.getOrElse(0.0); }
          }
          return sum;
        };
      default:
        throw new IllegalArgumentException("Objective type not implemented: " + type);
    }
  }

  /** {}@inheritDoc} */
  @Override
  public SlotfillPostProcessor and(final SlotfillPostProcessor alsoProcess) {
    if (alsoProcess instanceof HeuristicSlotfillPostProcessor) {
      final HeuristicSlotfillPostProcessor hpp = (HeuristicSlotfillPostProcessor) alsoProcess;
      final HeuristicSlotfillPostProcessor outer = this;
      return new HeuristicSlotfillPostProcessor() {
        @Override
        public Maybe<KBPSlotFill> isValidSlotAndRewrite(KBPEntity pivot, KBPSlotFill candidate) {
          Maybe<KBPSlotFill> outerValid = outer.isValidSlotAndRewrite(pivot, candidate);
          if (outerValid.isDefined()) {
            Maybe<KBPSlotFill> hppValid = hpp.isValidSlotAndRewrite(pivot, outerValid.get());
            if (hppValid.isDefined()) {
              return hppValid;
            } else {
              return Maybe.Nothing();

            }
          } else {
            return Maybe.Nothing();
          }
        }
        @Override
        public boolean pairwiseKeepLowerScoringFill(KBPEntity pivot, KBPSlotFill higherScoring, KBPSlotFill lowerScoring) {
          return outer.pairwiseKeepLowerScoringFill(pivot, higherScoring, lowerScoring) &&
              hpp.pairwiseKeepLowerScoringFill(pivot, higherScoring, lowerScoring);
        }
        @Override
        public boolean leaveOneOutKeepHeldOutSlot(KBPEntity pivot, IdentityHashSet<KBPSlotFill> others, KBPSlotFill candidate) {
          return outer.leaveOneOutKeepHeldOutSlot(pivot, others, candidate) &&
              hpp.leaveOneOutKeepHeldOutSlot(pivot, others, candidate);
        }
      };
    } else {
      return super.and(alsoProcess);
    }
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public List<KBPSlotFill> postProcess(KBPEntity pivot, List<KBPSlotFill> slotFills, GoldResponseSet checklist) {
    List<KBPSlotFill> filter1 = filterStep(pivot, slotFills, checklist);
    return filterStep(pivot, filter1, checklist);
  }

  /**
   * Independently of other slot fills, determine if this is even a reasonable slot fill,
   * and if it is, optionally rewrite it.
   * @param pivot The entity we are filling slots for
   * @param candidate The candidate slot fill
   * @return either Maybe.Nothing to ignore the slot, or a rewriten slot.
   */
  public abstract Maybe<KBPSlotFill> isValidSlotAndRewrite(KBPEntity pivot, KBPSlotFill candidate);

  /**
   * Filter a slot if it is directly inconsistent with another slot
   * @param pivot The entity we are filling slots for
   * @param higherScoring The higher scoring slot (this one always survives!)
   * @param lowerScoring The lower scoring slot (this is the one that may die)
   * @return True if the lower scoring slot should also be kept (this is the usual case)
   */
  public abstract boolean pairwiseKeepLowerScoringFill(KBPEntity pivot, KBPSlotFill higherScoring, KBPSlotFill lowerScoring);

  /**
   * Filter a slot if it is directly inconsistent with the set of other slots.
   * @param pivot The entity we are filling slots for
   * @param others The set of slots which are already filled.
   * @param candidate The slot which we are proposing to add
   * @return True if the candidate slot should also be kept (this is the usual case)
   */
  public abstract boolean leaveOneOutKeepHeldOutSlot(KBPEntity pivot, IdentityHashSet<KBPSlotFill> others, KBPSlotFill candidate);

  /**
   * A default implementation (effectively a NOOP) so that selective methods can be overwritten
   */
  public static class Default extends HeuristicSlotfillPostProcessor {
    @Override
    public Maybe<KBPSlotFill> isValidSlotAndRewrite(KBPEntity pivot, KBPSlotFill candidate) {
      return Maybe.Just(candidate);
    }
    @Override
    public boolean pairwiseKeepLowerScoringFill(KBPEntity pivot, KBPSlotFill higherScoring, KBPSlotFill lowerScoring) {
      return true;
    }
    @Override
    public boolean leaveOneOutKeepHeldOutSlot(KBPEntity pivot, IdentityHashSet<KBPSlotFill> others, KBPSlotFill candidate) {
      return true;
    }
  }

}
