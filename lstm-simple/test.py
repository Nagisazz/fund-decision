from FundCodeCrawler import FundCodeCrawler
from FundInfoCrawler import FundInfoCrawler
from FundPredictor import FundPredictor

funds = {
  "白酒" : {
    "page" : 3,
    "code" : '161725'
  }
}
for plateName,value in funds.items():
    print(plateName,value)
    # fundCodeCraw = FundCodeCrawler()
    # plateCode = fundCodeCraw.craw_page(plateName,value['page'])

    # fundInfoCraw = FundInfoCrawler(plateCode,value['code'])
    # fundInfoCraw.getFundTrain()
    # fundInfoCraw.getFundTest()

    # fundPredictor = FundPredictor(plateCode)
    fundPredictor = FundPredictor('801125')

    fundPredictor.predictor(0,1,True)

    # for i in range(0,5,2):
    #   fundPredictor.predictor(i/10,1,True)

    # for i in range(0,5,2):
    #   fundPredictor.predictor(i/10,2,True)
    
    # for i in range(0,5,2):
    #   fundPredictor.predictor(i/10,3,True)