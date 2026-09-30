// =======================================================
// PART 15. EXPORT TRAINING SAMPLES FOR GROUPED CROSS-VALIDATION
// Append to the end of the main UTCI script (after Ta_coarse, RH_coarse,
// predictorsTa/RH and bandsTa/RH are defined).
// Exports the exact Random Forest training samples (same sampling, seed,
// predictors and targets as downscaleVariable) with their location and
// parent ERA5-Land cell. The cross-validation is run offline with
// python/grouped_cv.py, which avoids the Earth Engine time limit.
// After Run: open the Tasks tab and start the two GroupedCV_Samples exports.
// =======================================================
var cvTag = (typeof exportTag !== 'undefined') ? exportTag :
  ((typeof cityName !== 'undefined' ? cityName : 'City') + '_' +
   (typeof dateString !== 'undefined' ? dateString.replace(/-/g, '') : 'date'));
var cvFolder = (typeof outFolder !== 'undefined') ? outFolder : 'GEE_UTCI_GroupedCV';

function exportTrainingSamples(targetImg, targetName, predictorStack, predictorBands) {
  var fc = predictorStack
    .select(predictorBands)
    .addBands(targetImg.rename(targetName))
    .sample({
      region: macroRegion.geometry(),
      scale: sampleScale,
      numPixels: sampleN,
      seed: seed,
      geometries: true,
      dropNulls: true,
      tileScale: 16
    })
    .map(function (f) {
      var xy = f.geometry().coordinates();
      var lon = ee.Number(xy.get(0));
      var lat = ee.Number(xy.get(1));
      var cell = lon.divide(0.1).round().multiply(100000).add(lat.divide(0.1).round());
      return f.set({lon: lon, lat: lat, era5_cell: cell}).setGeometry(null);
    });

  Export.table.toDrive({
    collection: fc,
    description: 'GroupedCV_Samples_' + targetName + '_' + cvTag,
    folder: cvFolder,
    fileNamePrefix: 'GroupedCV_Samples_' + targetName + '_' + cvTag,
    fileFormat: 'CSV'
  });
}

exportTrainingSamples(Ta_coarse, 'Ta', predictorsTa, bandsTa);
exportTrainingSamples(RH_coarse, 'RH', predictorsRH, bandsRH);
