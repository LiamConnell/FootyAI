# Footy RL

## Motivation

The rise of LLMs has completely taken over career, but I've grown miss the world of classical ML and RL. I wanted to refresh my memory on some of the fundamentals and take on the challenge of designing a RL environment from scratch. I also wanted to practice greenfield project development with AI assisted tools, seeing how much they could accelerate my pace of development. 

I found this to be a rewarding project with fun results. It supported my belief that experienced developers can use AI coding tools as a force multiplier rather than atrophy-inducing crutch. 

## Project Summary

<div style="text-align: center;">
    <video width="400" controls>
        <source src="./5000.mp4" type="video/mp4">
    </video>
</div>


This project tackles a multi-agent adversarial reinforcement learning problem by simulating a competitive soccer environment. In this setting, two teams of agents engage in a dynamic game where each player controls movement and kicking actions with the goal of scoring and defending, all while navigating continuous state and action spaces. The agents learn through self-play, contending with both cooperative strategies among teammates and adversarial tactics from opponents. 

The gameplay environment was build specially for this project and is completely novel. It features simple ball physics, boundary constraints and game rules. It takes place in continuous space over discrete time intervals. 

A reward signal shapes the behavior of agents by rewarding goal scoring. It optionally can also reward proximity to the ball and posession (kicks). 


