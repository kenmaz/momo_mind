require 'opencv'
include OpenCV

if ARGV.length < 2
  puts "Usage: ruby #{__FILE__} source dest"
  exit
end

input_file = ARGV[0]
output_file = ARGV[1]

#data = './data/haarcascades/haarcascade_frontalface_alt.xml'
data = 'cascade_fix.xml'

detector = CvHaarClassifierCascade::load(data)
image = CvMat.load(input_file)
detector.detect_objects(image).each do |region|
  puts region.inspect
  color = CvColor::Blue
  image.rectangle! region.top_left, region.bottom_right, :color => color
end

image.save_image(output_file)
#window = GUI::Window.new('Face detection')
#window.show(image)
#GUI::wait_key
