require 'net/http'
require 'json'
require 'open-uri'
require 'digest/md5'

#
# keywordでbing画像検索し画像のURLを取得しcsvを出力します
# csv:
#  <keyword>,<url>,<url-md5_hash>
#

APIKEY = '*** fixme ***'

def get(keyword, skip = 0)
  url = "https://api.datamarket.azure.com/Bing/Search/v1/Composite?Sources=%27image%27&Query=#{query(keyword)}&Market=%27ja-JP%27&Adult=%27Strict%27&ImageFilters=%27Size%3ALarge%27&$format=json&$skip=#{skip}"

  uri = URI(url)
  req = Net::HTTP::Get.new(uri.request_uri)
  req.basic_auth('', APIKEY)
  res = Net::HTTP.start(uri.hostname, uri.port, :use_ssl => uri.scheme == 'https') do |http|
    http.request(req)
  end
  body = JSON.parse(res.body, :symbolize_names => true)
  body[:d][:results][0][:Image].each do |page|
    media_url = page[:MediaUrl]
    md5 = Digest::MD5.hexdigest(media_url)
    puts "#{keyword},#{media_url},#{md5}"
  end

  sleep 5
end

def query(search_term)
  return URI.encode_www_form_component('\'' + search_term + '\'')
end

[100,200,300].each do |skip|
  #get('ももクロ', skip)
  #get('ももいろクローバーZ', skip)
  get('百田夏菜子', skip)
  get('玉井詩織', skip)
  get('佐々木彩夏', skip)
  get('有安杏果', skip)
  get('高城れに', skip)
  #get('川上アキラ', skip)
  #get('古屋智美', skip)
end
