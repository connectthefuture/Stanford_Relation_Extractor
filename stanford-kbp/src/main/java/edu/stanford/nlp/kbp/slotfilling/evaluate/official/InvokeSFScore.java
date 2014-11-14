package edu.stanford.nlp.kbp.slotfilling.evaluate.official;

import edu.stanford.nlp.kbp.common.Maybe;
import edu.stanford.nlp.kbp.slotfilling.evaluate.KBPScore;
import edu.stanford.nlp.util.logging.Redwood;

import java.io.*;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Modifier;
import java.net.URL;
import java.net.URLConnection;
import java.util.regex.*;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * A simple utility to invoke an official scorer from code, setting up the classpath, etc.
 *
 * @author Gabor Angeli
 */
public class InvokeSFScore {

  private static Redwood.RedwoodChannels logger = Redwood.channels("SFScore");
  private static Pattern  precision = Pattern.compile("^Precision:[^=]*= (.*)$");
  private static Pattern  recall    = Pattern.compile("^Recall:[^=]*= (.*).*");


  private static class CustomClassLoader extends ClassLoader {
    public Class loadClass(String name) throws ClassNotFoundException {
      if (!name.startsWith(InvokeSFScore.class.getPackage().getName())) {
        return super.loadClass(name);
      }
      try {
        URL url = super.getResource(name.replaceAll("\\.", File.separator) + ".class");
        System.out.println(name.replaceAll("\\.", File.separator) + ".class");
        assert url != null;
        URLConnection connection = url.openConnection();
        InputStream input = connection.getInputStream();
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        int data = input.read();
        while(data != -1){
          buffer.write(data);
          data = input.read();
        }
        input.close();
        byte[] classData = buffer.toByteArray();
        return defineClass(name, classData, 0, classData.length);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }
  }

  public static Maybe<KBPScore> scrapeScore(String output) {
    try {
      Double p = null;
      Double r = null;
      for (String line : output.split("[\\r\\n]+\\s*")) {
        Matcher prm = precision.matcher(line);
        if (prm.matches()) {
          p = Double.parseDouble(prm.group(1));
        }
        Matcher rcm = recall.matcher(line);
        if (rcm.matches()) {
          r = Double.parseDouble(rcm.group(1));
        }
      }
      if (r != null && p != null) {
        return Maybe.Just(new KBPScore(p, r, 0.0, new double[]{p}, new double[]{r}));
      } else {
        logger.err("Could not find P/R in output file");
        return Maybe.Nothing();
      }
    } catch (Throwable e) {
      logger.err(e);
    }
    return Maybe.Nothing();
  }

  /**
   * Directly call the main method on the scorer, without invoking a new JVM.
   * This will not work (generally) in multithreaded environments.
   */
  public static Maybe<KBPScore> invokeDirectly(Class sfScorer, File responseFile, File keyFile, File slotsFile) {
    String[] args = new String[]{ responseFile.getPath(), keyFile.getPath(), "slots=" + slotsFile.getPath(), "anydoc", "ignoreoffsets" };
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    PrintStream ps = new PrintStream(baos);
    System.setSecurityManager(new SecurityManager(){
      @Override public void checkExit(int status) {
        throw new RuntimeException("System attempted to exit with exit code: " + status);
      }
    });
    try {
      // Call the main method
      //noinspection unchecked
      ClassLoader loader = new CustomClassLoader();
      Class<?> scorer;
      synchronized (System.class) {
        System.setOut(ps);
        scorer = loader.loadClass(sfScorer.getCanonicalName());
      }
      scorer.getMethod("main", String[].class).invoke(null, (Object) args);
      // Try to clear the static fields of the class
      for (Field field : scorer.getDeclaredFields()) {
        if (Modifier.isStatic(field.getModifiers()) && !Modifier.isFinal(field.getModifiers()) &&
            !field.getType().isPrimitive()) {
          field.setAccessible(true);
          field.set(null, null);
        }
      }
      // Restore things
      String output = baos.toString("utf-8");
      baos.close();
      return scrapeScore(output);
    } catch (NoSuchMethodException | IllegalAccessException | InvocationTargetException | ClassNotFoundException | IOException e) {
      throw new RuntimeException(e);
    }
  }


  @SuppressWarnings("UnusedDeclaration")
  public static Maybe<KBPScore> invokeAsProcess(Class sfScorer, File responseFile, File keyFile, File slotsFile, boolean verbose) {
    String classpath = System.getProperties().getProperty("java.class.path");
    String classname = sfScorer.getCanonicalName();

    try {
      startTrack("Running " + sfScorer.getSimpleName());
      // Start process
      Process process = new ProcessBuilder("java", "-client", "-cp", classpath, "-mx256M", classname,
          responseFile.getPath(), keyFile.getPath(), "slots=" + slotsFile.getPath(),
          "anydoc", "ignoreoffsets").start();

      // Scrape output
      StringBuilder stdout = new StringBuilder();
      String line;
      BufferedReader br = new BufferedReader(new InputStreamReader(process.getInputStream()));
      while ((line = br.readLine()) != null) {
        if (verbose) { logger.log(line); }
        stdout.append(line).append("\n");
      }

      // Print error
      br = new BufferedReader(new InputStreamReader(process.getErrorStream()));
      while ((line = br.readLine()) != null) {
        if (verbose) { logger.err(line); }
      }

      // Wait for finish
      if (process.waitFor() == 0) {
        return scrapeScore(stdout.toString());
      } else {
        logger.warn("official scorer exited with non-zero exit code");
        return Maybe.Nothing();
      }
    } catch (IOException | InterruptedException e) {
      logger.err(e);
      return Maybe.Nothing();
    } finally {
      endTrack("Running " + sfScorer.getSimpleName());
    }
  }

  public static Maybe<KBPScore> invoke(Class sfScorer, File responseFile, File keyFile, File slotsFile) {
    return invokeAsProcess(sfScorer, responseFile, keyFile, slotsFile, false);
  }

}
