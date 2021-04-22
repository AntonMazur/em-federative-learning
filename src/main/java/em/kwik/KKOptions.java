/*
 * Created on Dec 2, 2010
 */
package em.kwik;

/**
 * Class to hold options for KK computing engine.
 * @author peter
 */
public class KKOptions implements Cloneable {
  
  static final int DEFAULT_NOISE_POINT = 1;
  static final double DEFAULT_DIST_THRESHOLD = Math.log(1000);
  static final double DEFAULT_FRAC_POINTS_CHANGED = 0.05;
  static final int DEFAULT_FULL_STEP_EVERY = 10;
  static final int DEFAULT_MAX_ITER = 500;
  static final int DEFAULT_SPLIT_EVERY = 40;
  static final double DEFAULT_PENALTY_MIX = 1.0;

  private int noisePoint;
  private double distThreshold;
  private double fracPointsChangedThreshold;
  private int fullStepEvery;
  private int maxIter;
  private int splitEvery;
  private boolean saveModel;
  private boolean distDump;
  private double penaltyMix;
  
  public KKOptions() {
    noisePoint = DEFAULT_NOISE_POINT;
    distThreshold = DEFAULT_DIST_THRESHOLD;
    fracPointsChangedThreshold = DEFAULT_FRAC_POINTS_CHANGED;
    fullStepEvery = DEFAULT_FULL_STEP_EVERY;
    maxIter = DEFAULT_MAX_ITER;
    splitEvery = DEFAULT_SPLIT_EVERY;
    penaltyMix = DEFAULT_PENALTY_MIX;
  }
  
  public Object clone() {
    KKOptions res = new KKOptions();
    res.noisePoint = this.noisePoint;
    res.distThreshold = this.distThreshold;
    res.fracPointsChangedThreshold = this.fracPointsChangedThreshold;
    res.maxIter = this.maxIter;
    res.splitEvery = this.splitEvery;
    res.saveModel = this.saveModel;
    res.distDump = this.distDump;
    res.penaltyMix = this.penaltyMix;
    return res;
  }

  public int getNoisePoint() {
    return noisePoint;
  }

  public void setNoisePoint(int noisePoint) {
    this.noisePoint = noisePoint;
  }

  public double getDistThreshold() {
    return distThreshold;
  }

  public void setDistThreshold(double distThreshold) {
    this.distThreshold = distThreshold;
  }

  public double getFracPointsChangedThreshold() {
    return fracPointsChangedThreshold;
  }

  public void setFracPointsChangedThreshold(double fracPointsChangedThreshold) {
    this.fracPointsChangedThreshold = fracPointsChangedThreshold;
  }

  public int getFullStepEvery() {
    return fullStepEvery;
  }

  public void setFullStepEvery(int fullStepEvery) {
    this.fullStepEvery = fullStepEvery;
  }

  public int getMaxIter() {
    return maxIter;
  }

  public void setMaxIter(int maxIter) {
    this.maxIter = maxIter;
  }

  public int getSplitEvery() {
    return splitEvery;
  }

  public void setSplitEvery(int splitEvery) {
    this.splitEvery = splitEvery;
  }

  public boolean isSaveModel() {
    return saveModel;
  }

  public void setSaveModel(boolean saveModel) {
    this.saveModel = saveModel;
  }

  public boolean isDistDump() {
    return distDump;
  }

  public void setDistDump(boolean distDump) {
    this.distDump = distDump;
  }
  
  /**
   * Get amount of BIC to use for penalty.
   * @return
   */
  public double getPenaltyMix() {
    return penaltyMix;
  }
  
  /**
   * Set amount of BIC to use for penalty, must be between 0 and 1
   * @param pmix
   */
  public void setPenaltyMix(double pmix) {
    if (pmix<0.0 | pmix >1.0) {
      throw new IllegalArgumentException("penalty mix must be between 0 and 1");
    }
    this.penaltyMix = pmix;
  }
}