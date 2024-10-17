
"use strict";

let Keypoint2Di = require('./Keypoint2Di.js');
let BoundingBox2Di = require('./BoundingBox2Di.js');
let RGBDSensors = require('./RGBDSensors.js');
let Object = require('./Object.js');
let BoundingBox2Df = require('./BoundingBox2Df.js');
let PosTrackStatus = require('./PosTrackStatus.js');
let Skeleton2D = require('./Skeleton2D.js');
let BoundingBox3D = require('./BoundingBox3D.js');
let ObjectsStamped = require('./ObjectsStamped.js');
let Skeleton3D = require('./Skeleton3D.js');
let Keypoint2Df = require('./Keypoint2Df.js');
let Keypoint3D = require('./Keypoint3D.js');
let PlaneStamped = require('./PlaneStamped.js');

module.exports = {
  Keypoint2Di: Keypoint2Di,
  BoundingBox2Di: BoundingBox2Di,
  RGBDSensors: RGBDSensors,
  Object: Object,
  BoundingBox2Df: BoundingBox2Df,
  PosTrackStatus: PosTrackStatus,
  Skeleton2D: Skeleton2D,
  BoundingBox3D: BoundingBox3D,
  ObjectsStamped: ObjectsStamped,
  Skeleton3D: Skeleton3D,
  Keypoint2Df: Keypoint2Df,
  Keypoint3D: Keypoint3D,
  PlaneStamped: PlaneStamped,
};
