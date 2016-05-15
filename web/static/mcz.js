$(function() {
  //var dropzone = new Dropzone("#upload_form");
  /**
  */
  Dropzone.options.uploadForm = {
    maxFiles: 1,
    accept: function(file, done) {
      done();
    },
    init: function() {
      var statusLabel = $("#status");

      this.on("addedfile", function() {
        if (this.files[1] != null) {
          this.removeFile(this.files[0]);
        }
      });
      this.on("processing", function(file) {
        console.log("processing");
        $("#status").text('processing')

        var table = $("#results");
        table.empty();
      })
      this.on("uploadprogress", function(file, progress, bytesSent) {
        console.log("uploadprogress");
        $("#status").text('uploadprogress:'+progress);
        if (progress == 100) {
          $("#status").text('prosessing...');
        }
      })
      this.on("sending", function(file) {
        console.log("sending");
      })
      this.on("success", function(file, response) {
        $("#status").text('success')
        console.log("success");
        console.log(response);

        var table = $("#results");

        var results = response['results'];
        if (results.length == 0) {
          $("#status").text('顔を検知出来ませんでした')

        } else {
          $(results).each(function(idx, item) {
            console.log(item);
            var tr = $("#results").append($("<tr></tr>"));
            tr.append($("<td></td>")).append(
              $("<img></img>").attr('src', item['file'])
            );
            var ul = tr.append('<ul></ul>');
            $(item['rank']).each(function(j, member) {
              ul.append($('<li></li>').text(member["name"] + ':' + member["rate"] + '%'));
            });
          });
        }
      })
      this.on("error", function(file) {
        $("#status").text('error')
        console.log("error");
      })
    }
  }
});
