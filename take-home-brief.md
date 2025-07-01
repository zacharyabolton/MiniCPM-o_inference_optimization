# Outspeed MLE Take-Home

1. Background
   1. [**MiniCPM-o**](https://github.com/OpenBMB/MiniCPM-o) is the latest
      series of end-side multimodal LLMs (MLLMs) and can take images, video,
      text, and audio as inputs and provide high-quality text and speech
      outputs in an end-to-end fashion.
   2. With a total of 8B parameters, this end-to-end model
        **achieves comparable performance to GPT-4o-202405 in vision, speech, and
      multimodal live streaming**, making it one of the most versatile and
      performant models in the open-source community.
2. Objectives
   1. Your goal is to profile and optimize the performance of a working
      realtime version of MiniCPM-o on [Modal](https://modal.com/).
      1. Current MiniCPM performance on A10G:
         1. Time to first byte = 2s
         2. Realtime factor = 1.1x
      2. Desired MiniCPM performance on A10G:
         1. Time to first byte = 1s
         2. Realtime factor = 0.5x
   2. Your submission will be judged on the following criteria:
      1. Problem solving: The approach you use to solve the problem is as
         important as the solution. Clearly outline and document the steps you
         will take to identify, debug and test the problem.
      2. Completeness: The submission should run without any errors and
         produce similar quality audio as the unoptimized version.
      3. Documentation: Document all approaches, experiments, and findings
         clearly, including details on methodologies, results, and insights
         gained in a text file in the Github repo.
3. Guidelines
   1. Start by creating a fresh github repo with all the uncompressed code
      (download link given below). You should push changes and updates to this
      repo.
      1. Add `janak2` and `altairmn` as collaborators so we can check your
         submission.
   2. The take home is intended to be completed in 2 hour.
   3. The problem is not intended to involve a lot of coding. If you have to
      write more than 100 lines of code, you probably are not on the right
      track.
4. Code:
   1. Please download and unzip the following file:

      [mle-take-home.zip](Outspeed%20MLE%20Take-Home%20223c491cb1a48004ba46c2e2348f6098/mle-take-home.zip)

   2. The instructions can be found in README.md in the folder.
