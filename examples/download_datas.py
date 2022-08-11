import os
from openiva.commons.io import download_file

if __name__=="__main__":
    


    urls=["https://s3.openi.org.cn/opendata/attachment/4/4/4425761f-e44e-4ed9-b2a4-a5c2469aa0fa?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20211227%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20211227T144237Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22arc_mbv2_ccrop_sim.onnx%22&X-Amz-Signature=4cfe31cc9a64f735d4f9dc8f3d3e77cfcb69ded298302821c30bc4ca602da678",\
        "https://s3.openi.org.cn/opendata/attachment/2/7/277df3cb-983f-4365-977a-c55239ab8b7b?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20211227%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20211227T143600Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22yolox_m_sim.onnx%22&X-Amz-Signature=98d6af4d7c73a1aa4986b186d185cb3260ab50957d9f06eda9f7ecc7490e0dd0",\
        "https://s3.openi.org.cn/opendata/attachment/2/4/24024313-fddd-4b78-b968-41353c0b3798?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20211227%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20211227T144732Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22yolox_s_sim.onnx%22&X-Amz-Signature=fa18fda4027616f0a26b7e041a0647408a986582585d47aeeee2079d55dbd484",\
        "https://s3.openi.org.cn/opendata/attachment/9/e/9edeedf9-99f5-48e1-a934-97c3554ad9b6?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20211227%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20211227T144305Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22yolox_tiny_sim.onnx%22&X-Amz-Signature=ff3199140012f462a529ccb62c14aa0ae12ed325bb97607a35e9e7827eed856e",
        "https://s3.openi.org.cn/opendata/attachment/f/0/f0578636-d9f1-48e9-a3e8-3561150bff2b?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20211227%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20211227T140542Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22inception_clip.mp4%22&X-Amz-Signature=9c9bffed6c747f7fd2395ae4d4eca6d1c295e8cd0d5127435c0542f920260843",\
        "https://s3.openi.org.cn/opendata/attachment/d/7/d733ad56-033b-4d70-a341-21ec04607f7a?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20211228%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20211228T095434Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22SourceHanSansHWSC-Regular.otf%22&X-Amz-Signature=3a62b36f8ad139c2aa79e5b47fd050b21b8089aa9c2132830555b71d349b4275"]

    pathes=["weights/arc_mbv2_ccrop_sim.onnx",\
        "weights/yolox_m_sim.onnx",\
        "weights/yolox_s_sim.onnx",\
        "weights/yolox_tiny_sim.onnx",\
        "datas/videos_test/inception_clip.mp4",\
        "datas/fonts/SourceHanSansHWSC-Regular.otf"]

    md5s=["789aa5851d5f9d97e5123fe38dde4972",\
        "6c7a818c52ba73e078d95babfe462d89",\
        "92bbb8f4010bfde4f5fe78851b57a3c2",\
        "fb7d04a6e350e691308df7e764f7f9a5",\
        "5ccd01e21c5fab2933db33dc35b9877e",\
        "510d964928c73c37d83d0d670509dbc7"]
        
    if not os.path.exists("datas/videos_test"):
        os.mkdir("datas/videos_test")
    
    for url,path,md5 in zip(urls,pathes,md5s):
        download_file(url,path,md5)