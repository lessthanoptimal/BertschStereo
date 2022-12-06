```bash
./gradlew StereoDisparityApp
java -jar StereoDisparity.jar
java -jar StereoDisparity.jar -l "glob:left*.jpg" -r "glob:right*.jpg" -c stereo.yaml  -o results
```

```bash
./gradlew TrinocularDisparityApp
java -jar TrinocularDisparity.jar -i "path/to/data" --Left "glob:left*.jpg" --Right "glob:right*.jpg" --Right "glob:Middle*.jpg" -c calibration.yaml  -o results
```