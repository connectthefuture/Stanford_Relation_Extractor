package edu.stanford.nlp.kbp.slotfilling.evaluate.inference;

import edu.stanford.nlp.kbp.common.Pointer;
import edu.stanford.nlp.math.SloppyMath;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.ArrayIterable;
import edu.stanford.nlp.util.Execution;
import edu.stanford.nlp.util.Index;
import edu.stanford.nlp.util.logging.Redwood;

import java.util.*;

import static edu.stanford.nlp.util.logging.Redwood.Util.log;

/**
 * A very simple Bayes Net taking only binary variables.
 * Not on accident, this is exactly the type of net we need for inference.
 *
 * @author Gabor Angeli
 */
public class BayesNet<E> extends AbstractSet<BayesNet.Factor> {

  protected static interface Factor {
    public String getName();
    public double logProb(boolean[] assignment);
    public Collection<Integer> components();
  }

  private class AssignmentState {
    // TODO(arun): Make dependent on the number of variables
    public final double RESTART_ITERS = (int) 1e5; // Number of iterations after which to restart

    public boolean doMAP;
    public boolean doMarginal;

    private final Random rand;
    public boolean[] assignment;
    public double[] counts;
    public long[] lastUpdate;
    public double logScore;
    public long numIters = 0;

    public boolean[] bestAssignment;
    public double bestLogScore = Double.NEGATIVE_INFINITY;

    public AssignmentState(int seed) {
      this.rand = new Random(seed);
      this.assignment = new boolean[predicates.length];

      for(Map.Entry<Integer,Boolean> value : initialValues.entrySet())
        this.assignment[value.getKey()] = value.getValue();

      // Assign fixed values
      this.counts = new double[predicates.length];
      this.lastUpdate = new long[predicates.length];
      this.bestAssignment = new boolean[assignment.length];

      randomRestart();
      this.logScore = computeLogScore();
      assert !SloppyMath.isVeryDangerous(this.logScore);
    }

    private AssignmentState(int seed, boolean[] assignment) {
      this.rand = new Random(seed);
      this.assignment = assignment;
      this.counts = new double[predicates.length];
      this.bestAssignment = new boolean[assignment.length];
      this.logScore = computeLogScore();
      assert !SloppyMath.isVeryDangerous(this.logScore);
    }

    /**
     * Randomly reinitialize all the variables that are not fixed
     * TODO(arun): Use prior probabilities to set these
     */
    protected void randomRestart() {
      // Update counts
      if(doMarginal)
        updateCounts();
      for (int i = 0; i < assignment.length; ++i) {
        if(!isFixed[i]) {
          this.assignment[i] = rand.nextDouble() < priors[i];
        }
      }
      this.logScore = computeLogScore();
    }

    protected void updateCounts() {
      if(numIters == 0) return;
      for(int i = 0; i < assignment.length; i++) {
        this.counts[i] += ((this.assignment[i]?1.:0.) - this.counts[i]) * (numIters - lastUpdate[i])/(numIters);
        lastUpdate[i] = numIters;
      }
    }

    public void gibbsStep() {
      // Random restart every X steps
      if (numIters % RESTART_ITERS == 0)  randomRestart();

      // Mitigate floating point drift
      if(doMAP && (numIters % 10000 == 0)) {
        assert Double.isInfinite(logScore) || Math.abs(logScore - computeLogScore()) < 0.1;
        // Deterministically hill climb a bit
        if (doHillclimb) {
          for (int i = 0; i < adjustable.length; ++i) {
            gibbsStep(adjustable[i], true);
          }
        }
        // Compute empirical score
        assert Double.isInfinite(logScore) || Math.abs(logScore - computeLogScore()) < 0.1;
        this.logScore = computeLogScore();
      }

      // Chose the variable to flip
      int toFlip = adjustable[rand.nextInt(adjustable.length)];
      // Never pick a fixed value

      // Do gibbs step
      gibbsStep(toFlip, false);
      numIters += 1;

      // Update counts
      if(doMarginal)
        counts[toFlip] += ((assignment[toFlip] ? 1. : 0.) - counts[toFlip]) * (numIters - lastUpdate[toFlip])/numIters;
    }

    public void gibbsStep(int toFlip, boolean deterministicHillclimb) {
      // Never change a fixed value
      assert !isFixed[toFlip];

      // Compute the scores for each domain
      double scoreTrue;
      double scoreFalse;
      double tmpScore = this.logScore;
      if (assignment[toFlip]) {
        scoreTrue = tmpScore;
        for (Factor fact : factorsByPredicate[toFlip]) {
          assignment[toFlip] = true;
          tmpScore -= fact.logProb(assignment);
          assignment[toFlip] = false;
          tmpScore += fact.logProb(assignment);
        }
        scoreFalse = tmpScore;
      } else {
        scoreFalse = tmpScore;
        for (Factor fact : factorsByPredicate[toFlip]) {
          assignment[toFlip] = false;
          tmpScore -= fact.logProb(assignment);
          assignment[toFlip] = true;
          tmpScore += fact.logProb(assignment);
        }
        scoreTrue = tmpScore;
      }
      // Compute conditional probability
      double probTrue;
      if(Double.isInfinite(scoreTrue) && Double.isInfinite(scoreFalse))
        probTrue = 0.5;
      else if(Double.isInfinite(scoreTrue))
        probTrue = 1.0;
      else if(Double.isInfinite(scoreFalse))
        probTrue = 0.0;
      else {
        double denominator = SloppyMath.logAdd(scoreFalse, scoreTrue);
        probTrue = Math.exp(scoreTrue - denominator);
      }
      assert probTrue >= -1e-3 && probTrue <= 1 + 1e-3;
      if (probTrue < 0.0) { probTrue = 0; }
      if (probTrue > 1.0) { probTrue = 1.0; }

      // Do Gibbs Flip
      if (deterministicHillclimb) {
        if (probTrue > 0.5) {
          assignment[toFlip] = true;
          this.logScore = scoreTrue;
        } else {
          assignment[toFlip] = false;
          this.logScore = scoreFalse;
        }
      } else {
        if (rand.nextDouble() < probTrue) {
          assignment[toFlip] = true;
          this.logScore = scoreTrue;
        } else {
          assignment[toFlip] = false;
          this.logScore = scoreFalse;
        }
      }

      // keep track of best assignment
      if (doMAP && this.logScore > bestLogScore) {
        System.arraycopy(assignment, 0, bestAssignment, 0, assignment.length);
        bestLogScore = tmpScore;
      }
    }

    public double computeLogScore() {
      double logScore = 0.0;
      for (Factor factor : factors) {
        logScore += factor.logProb(assignment);
      }
      return logScore;
    }

    public String toString() {
      StringBuilder sb = new StringBuilder();
      sb.append(logScore).append(" ");
      for(int i = 0; i < assignment.length; i++)
      {
        if(!assignment[i]) sb.append('!');
        sb.append(predicates[i]).append(", ");
      }
      return sb.toString();
    }
  }

  protected final E[] predicates;
  protected final Factor[] factors;
  protected final Collection<Factor>[] factorsByPredicate;
  protected final Index<E> index;
  private final Map<Integer,Boolean> initialValues; // If a variable has a 'fixedValue' do not ever change it.
  private final boolean[] isFixed; // If a variable has a 'fixedValue' do not ever change it.
  private final double[] priors; // If a variable has a 'fixedValue' do not ever change it.
  protected final int[] adjustable;

  private final boolean doHillclimb;

  protected BayesNet(Index<E> index, E[] predicates, Factor[] factors,
                     Map<Integer,Double> priors, Map<Integer,Boolean> initialValues,
                     boolean doHillclimb) {
    this.index = index;
    this.predicates = predicates;
    this.factors = factors;
    this.priors = new double[predicates.length];
    for(int i = 0; i < predicates.length; i++) {
      this.priors[i] = priors.containsKey(i) ? Math.exp(priors.get(i)) : 0.2; // Really, low priors.
    }

    this.initialValues = initialValues;
    this.isFixed = new boolean[predicates.length];
    for(Map.Entry<Integer,Boolean> value : initialValues.entrySet())
      this.isFixed[value.getKey()] = true;
    this.adjustable = new int[predicates.length - initialValues.size()];
    {
      int j = 0;
      for(int i = 0; i < isFixed.length; i++ )
        if(!isFixed[i]) adjustable[j++] = i;
      assert j == adjustable.length;
    }

    this.doHillclimb = doHillclimb;
    //noinspection unchecked
    this.factorsByPredicate = new Collection[predicates.length];
    for (int i = 0; i < factorsByPredicate.length; ++i) {
      factorsByPredicate[i] = new ArrayList<>();
    }
    for (Factor factor : factors) {
      for (int component : factor.components()) {
        factorsByPredicate[component].add(factor);
      }
    }
  }

  @Override
  public Iterator<Factor> iterator() { return new ArrayIterable<>(factors).iterator(); }
  @Override
  public int size() { return factors.length; }

  public int variableCount() { return predicates.length; }

  public Counter<E> gibbsMLE(final int numIters) {
    Counter<E> trueFacts = new ClassicCounter<>();
    for (Map.Entry<E, Double> entry : gibbsMarginals(numIters).entrySet()) {
      if (entry.getValue() > 0.5) { trueFacts.setCount(entry.getKey(), entry.getValue()); }
    }
    return trueFacts;
  }

  public Counter<E> gibbsMAP(final int numIters) {
    if(adjustable.length == 0) {
      log("Warning: no-non-fixed predicates in BayesNet;");
      // Just return initialValues
      Counter<E> assignment = new ClassicCounter<>();
      for(Map.Entry<Integer, Boolean> entry : initialValues.entrySet())
        assignment.setCount(predicates[entry.getKey()], entry.getValue() ? 1.0 : 0.0 );
      return assignment;
    }

    // Sample
    Collection<Runnable> threads = new ArrayList<>();
    final boolean[] bestAssignment = new boolean[this.predicates.length];
    final Pointer<Double> bestLogScore = new Pointer<>(Double.NEGATIVE_INFINITY);
    for (int i = 0; i < Execution.threads; ++i) {
      final int seed = i;
      threads.add(() -> {
        AssignmentState state = new AssignmentState(seed);
        state.doMAP = true;
        for (int k = 0; k < numIters; ++k) {
          state.gibbsStep();
        }

        synchronized (bestLogScore) {
          if (bestLogScore.dereference().get() < state.bestLogScore) {
            bestLogScore.set(state.bestLogScore);
            System.arraycopy(state.bestAssignment, 0, bestAssignment, 0, state.bestAssignment.length);
          }
        }
      });
    }
    // Run (multithread)
    Redwood.Util.threadAndRun("Gibbs Sampling", threads, threads.size());
    // Translate Assignment
    log("Best assignment had log score: " + bestLogScore.dereference().getOrElse(-1.));
    Counter<E> assignment = new ClassicCounter<>();
    for (int i = 0; i < predicates.length; ++i) {
      if (bestAssignment[i]) {
        assignment.setCount(predicates[i], Double.POSITIVE_INFINITY);
      }
    }
    return assignment;
  }

  public Counter<E> gibbsMarginals(final int numIters) {
    if(adjustable.length == 0) {
      log("Warning: no-non-fixed predicates in BayesNet;");
      // Just return initialValues
      Counter<E> assignment = new ClassicCounter<>();
      for(Map.Entry<Integer, Boolean> entry : initialValues.entrySet())
        assignment.setCount(predicates[entry.getKey()], entry.getValue() ? 1.0 : 0.0 );
      return assignment;
    }
    // Sample
    Collection<Runnable> threads = new ArrayList<>();
    final Counter<E> marginals = new ClassicCounter<>();
    final Pointer<Double> bestLogScore = new Pointer<>(Double.NEGATIVE_INFINITY);
    for (int i = 0; i < Execution.threads; ++i) {
      final int seed = i;
      threads.add(() -> {
        AssignmentState state = new AssignmentState(seed);
        state.doMarginal = true;
        for (int k = 0; k < numIters; ++k) {
          state.gibbsStep();
        }
        state.updateCounts();

        synchronized (bestLogScore) {
          if (bestLogScore.dereference().get() < state.bestLogScore) {
            bestLogScore.set(state.bestLogScore);
          }
          for(int k = 0; k < predicates.length; k++)
            marginals.incrementCount(predicates[k], state.counts[k] / Execution.threads);
        }
      });
    }
    // Run (multithread)
    Redwood.Util.threadAndRun("Gibbs Sampling", threads, threads.size());
    return marginals;
  }

  /**
   * Return the log probability of a given assignment.
   * Entries which are not provided in the given assignment set are considered false by default.
   * @param truePredicates The set of predicates to assign as true; predicates not in this set are considered false
   * @return The log probability of the assignment
   */
  public double logProb(Set<E> truePredicates) {
    boolean[] assignmentArray = new boolean[predicates.length];
    for (int i = 0; i < assignmentArray.length; ++i) {
      assignmentArray[i] = truePredicates.contains(predicates[i]);
    }
    return new AssignmentState(42, assignmentArray).computeLogScore();
  }

}
