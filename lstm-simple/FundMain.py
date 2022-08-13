from FundCodeCrawler import FundCodeCrawler
from FundInfoCrawler import FundInfoCrawler
from FundPredictor import FundPredictor

fundCodeCraw = FundCodeCrawler()
plateCode = fundCodeCraw.craw_page('军工',5)

fundInfoCraw = FundInfoCrawler(plateCode,'001838')
fundInfoCraw.getFundTrain()
fundInfoCraw.getFundTest()

fundPredictor = FundPredictor(plateCode)
fundPredictor.predictor()