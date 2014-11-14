package edu.stanford.nlp.kbp.entitylinking;

import java.io.Serializable;
import java.util.List;

/**
 * TODO(gabor) JavaDoc
 *
 * @author Gabor Angeli
 */
public class ConditionalFeature implements Feature {

  public static class Specification implements Serializable {
    public final Class<? extends Feature> featureClass;
    public final List<String> conditions;
    public Specification(List<String> conditions, Class<? extends Feature> featureClass) {
      this.featureClass = featureClass;
      this.conditions = conditions;
    }
    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (!(o instanceof Specification)) return false;
      Specification that = (Specification) o;
      return conditions.equals(that.conditions) && featureClass.equals(that.featureClass);
    }
    @Override
    public int hashCode() {
      int result = featureClass.hashCode();
      result = 31 * result + conditions.hashCode();
      return result;
    }
  }

  public final Feature feature;
  public final String condition;

  public ConditionalFeature(String condition, Feature feature) {
    this.feature = feature;
    this.condition = condition;
  }


  @Override
  public double getCount() { return feature.getCount(); }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (!(o instanceof ConditionalFeature)) return false;
    ConditionalFeature that = (ConditionalFeature) o;
    return condition.equals(that.condition) && feature.equals(that.feature);

  }

  @Override
  public int hashCode() {
    int result = feature.hashCode();
    result = 31 * result + condition.hashCode();
    return result;
  }

  @Override
  public String toString() {
    return condition + "::" + feature.toString();
  }
}
