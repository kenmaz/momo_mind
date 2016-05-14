$(function() {
  Dropzone.options.myAwesomeDropzone = {
    accept: function(file, done) {
      alert("wee");
      done();
    }
  }
});
