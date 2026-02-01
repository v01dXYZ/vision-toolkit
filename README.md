# vision-toolkit

![tarsier](./assets/crop_tarsier_2.jpg)


> **Check the Docs!** <!-- FOR_README_ONLY -->
> <!-- FOR_README_ONLY -->
> You can check the docs at https://v01dxyz.github.io/vision-toolkit. <!-- FOR_README_ONLY -->


## Summary

`vision-toolkit` is a Python library for processing eye tracking data and extract from them:

- **Scanpaths**: A macro trajectory of the gaze without taking into account 
    all the little movements of the eyes around what they see
- **Areas of Interest**: Zones the viewer is attracted to.

Below a video showing the gaze of multiple viewers watching a movie scene:

![hollywood2-eye-tracking-example](./assets/hollywood2.gif)

`vision-toolkit` supports:

* multiple kind of coordinates:
  
  - 2D: representing a point on a screen. Coordinates could be cartesian or angular.
  - 3D: an estimation of where both eyes gazes converge, a.k.a. point of gaze
* multiple kind of frames:

  - static frame (the head stays still, e.g. medical settings)
  -  dynamic frame (the head moves, e.g. glasses or VR sets)
* using data from multiple viewers to analyse the same scene or event

## Credits

This work was done by Quentin Laborde as an extension to his PhD and
Robert Dazi for industrializing the package development, both as part
of Centre Borelli from ENS Paris-Saclay.
