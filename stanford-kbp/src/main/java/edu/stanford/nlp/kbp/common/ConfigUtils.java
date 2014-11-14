package edu.stanford.nlp.kbp.common;

import com.typesafe.config.ConfigIncludeContext;
import com.typesafe.config.ConfigIncluder;
import com.typesafe.config.ConfigObject;
import com.typesafe.config.ConfigOrigin;
import com.typesafe.config.ConfigParseOptions;
import com.typesafe.config.ConfigParseable;
import com.typesafe.config.ConfigSyntax;

/**
 * Utility classes for the typesafe config system to allow for hybrid conf/properties files
 * Why doesn't the typesafe config system just handle this?
 *
 * @author Angel Chang
 */
public class ConfigUtils {

  public static ConfigParseOptions getParseOptions() {
    ConfigParseOptions configParseOptions = ConfigParseOptions.defaults();
    configParseOptions = configParseOptions.setIncluder(
            new CustomConfigIncluder( configParseOptions.getIncluder() ));
    return configParseOptions;
  }

  /**
   * Custom config includer that changes the config syntax of the file being
   * included depending on the file extension (instead of using the original
   *   config syntax).  Allows for hybrid of conf/json/properties files.
   */
  public static class CustomConfigIncluder implements ConfigIncluder {
    // Actual includer that does stuff
    private final ConfigIncluder includer;

    public CustomConfigIncluder(ConfigIncluder includer) {
      this.includer = includer;
    }

    @Override
    public ConfigIncluder withFallback(ConfigIncluder configIncluder) {
      return new CustomConfigIncluder(configIncluder);
    }

    @Override
    public ConfigObject include(ConfigIncludeContext configIncludeContext, String s) {
      ConfigSyntax syntax = null;
      if (s.endsWith("conf")) {
        syntax = ConfigSyntax.CONF;
      } else if (s.endsWith("json")) {
        syntax = ConfigSyntax.JSON;
      } else if (s.endsWith("properties")) {
        syntax = ConfigSyntax.PROPERTIES;
      }
      if (syntax != null) {
        ConfigParseOptions newParseOptions = configIncludeContext.parseOptions().setSyntax(syntax);
        return includer.include(new CustomConfigIncludeContext(configIncludeContext, newParseOptions), s);
      } else {
        // who knows what the syntax was suppose to be
        return includer.include(configIncludeContext, s);
      }
    }
  }

  public static class CustomConfigParseable implements ConfigParseable {
    // actual ConfigParseable that does stuff
    private final ConfigParseable configParseable;
    private final ConfigParseOptions parseOptions;

    public CustomConfigParseable(ConfigParseable configParseable, ConfigParseOptions parseOptions) {
      this.configParseable = configParseable;
      this.parseOptions = parseOptions;
    }

    @Override
    public ConfigObject parse(ConfigParseOptions options) {
      return configParseable.parse(options);
    }

    @Override
    public ConfigOrigin origin() {
      return configParseable.origin();
    }

    @Override
    public ConfigParseOptions options() {
      return parseOptions;
    }
  }

  public static class CustomConfigIncludeContext implements ConfigIncludeContext {
    // actual ConfigIncludeContext that does stuff
    private final ConfigIncludeContext configIncludeContext;
    private final ConfigParseOptions parseOptions;

    public CustomConfigIncludeContext(ConfigIncludeContext configIncludeContext, ConfigParseOptions parseOptions) {
      this.configIncludeContext = configIncludeContext;
      this.parseOptions = parseOptions;
    }

    @Override
    public ConfigParseable relativeTo(String s) {
      ConfigParseable parseable = configIncludeContext.relativeTo(s);
      assert parseable != null;
      // Here is where we fix the parseOptions....
      // since the SimpleIncluder's RelativeNameSource.nameToParseable disregards the passed in options
      return new CustomConfigParseable(parseable, parseOptions);
    }

    @Override
    public ConfigParseOptions parseOptions() {
      return parseOptions;
    }
  }


}
