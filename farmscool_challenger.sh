#!/bin/bash


#!/bin/sh

ProgName=$(basename $0)

ChallengeHome=/home/challenger
DataDir=/mnt/data/challenge

sub_help(){
    echo "Usage: $ProgName <subcommand> [options]"
    echo ""
    echo "Subcommands:"
    echo "    add               Add script to queue"
    echo "    log               View the logs of current job "
    echo "    leaderboard       Print current leaderboard"
    echo "    info              See queue"
    echo "    suspend           Suspend queue"
    echo "    resume            Resume queue"
    echo "    activate          Activate environment (for pip install)"
    echo "    play              Play manually."
    echo "    top               Display all jobs and kill them if necessary."
    echo "    "
    echo ""
    echo "For help with each subcommand run:"
    echo "$ProgName <subcommand> -h|--help"
    echo ""

}

sub_add(){
    case $1 in  "" | "-h" | "--help")
                   echo "Usage: $ProgName add <python_file.py> <n_fit>"
                   echo ""
                   echo "    Add your agent to the queue in the challenge."
                   echo ""
                   echo "    Parameters:"
                   echo "       python_file: a .py file containing a rlberry agent class called \"Agent\"."
                   echo "       n_fit: an int, number of steps to train for."
                   echo ""
                   echo "    Example:"
                   echo "       $ProgName add ppo.py 10000"
                   echo ""
                   echo "       where ppo.py contain the following:"
                   echo ""
                   echo "from rlberry.agents import PPOAgent"
                   echo "class Agent(PPOAgent):"
                   echo "    def __init__(self, env, **kwargs):"
                   echo "        PPOAgent.__init__(self, env, **kwargs)"
                   echo "        self.batch_size = 16"
                   echo "        self.gamma = 0.9"

                   ;;
                   *)
                       source $ChallengeHome/virtualenv/bin/activate
                       cp  $(realpath $1) $DataDir/scripts
                       sudo -u challenger python $ChallengeHome/rlberry-farms/challenge/add_xp_to_queue.py $(basename $1) $2 $(whoami)
                       ;;
     esac
}



sub_info(){
    case $1 in "-h" | "--help")
                   echo "Usage: $ProgName info"
                   echo ""
                   echo "    Print info on the current queue of processes in the challenge."
                   echo ""

                   ;;
                   *)
                       source $ChallengeHome/virtualenv/bin/activate
                       rq info
                       ;;
     esac
}



sub_play(){
    case $1 in "-h" | "--help")
                   echo "Usage: $ProgName play <farm_num>"
                   echo ""
                   echo "    Play to farm number <farm_num>."
                   echo ""

                   ;;
                   *)
                       source $ChallengeHome/virtualenv/bin/activate
                       python $ChallengeHome/rlberry-farms/examples/interactive_farm$1.py
                       ;;
     esac
}



sub_dashboard (){
    case $1 in "-h" | "--help")
                   echo "Usage: $ProgName dashboard"
                   echo ""
                   echo "    Launch dashboard app. Can be tunneled locally with (on your local linux computer)"
                   echo "    $ ssh -L 9181:127.0.0.1:9181 yourlogin@flanders.lille.inria.fr."
                   echo "    where 9181 is the appropriate port and yourlogin is your inria login"
                   echo ""

                   ;;
                   *)
                       source $ChallengeHome/virtualenv/bin/activate
                       rq-dashboard
                       ;;
     esac
}

sub_log (){
    case $1 in "-h" | "--help")
                   echo "Usage: $ProgName log"
                   echo ""
                   echo "    Print logs"
                   echo ""

                   ;;
                   *)
                       tail -f $ChallengeHome/rlberry-farms/challenge/logfile.log
                       ;;
     esac
}

sub_activate(){
    case $1 in "-h" | "--help")
                   echo "Usage: $ProgName activate"
                   echo ""
                   echo "    Activate the virtual environment used in the challenge. Be careful to not break it!"
                   echo ""

                   ;;
                   *)
                       /bin/bash -c ". /home/challenge_env/virtualenv/bin/activate; exec /bin/bash --rcfile /home/challenge_env/.bashrc -i"
                       ;;
     esac
}




sub_leaderboard(){
    case $1 in "-h" | "--help")
                   echo "Usage: $ProgName leaderboard [OPTIONS]"
                   echo ""
                   echo "    Print current leaderboard."
                   echo "  "
                   echo "    Options "
                   echo "      -a       print all scores "
                   echo "      -m       print all my scores "
                   echo ""

                   ;;
                *)
                       source $ChallengeHome/virtualenv/bin/activate
                       python $ChallengeHome/rlberry-farms/challenge/print_leaderboard.py $1
                       ;;
     esac
}


sub_suspend(){
    case $2 in "-h" | "--help")
                   echo "Usage: $ProgName suspend"
                   echo ""
                   echo "    Suspend the current queue of processes in the challenge."
                   echo ""

                   ;;
                *)
                    source $ChallengeHome/virtualenv/bin/activate
                    rq suspend
                    ;;
     esac
}

sub_resume(){
    case $1 in  "-h" | "--help")
                   echo "Usage: $ProgName resume"
                   echo ""
                   echo "   Resume the current queue of processes in the challenge."
                   echo ""

                   ;;
                *)
                    source $ChallengeHome/virtualenv/bin/activate
                    rq resume
                    ;;
     esac
}

sub_top(){
    case $1 in  "-h" | "--help")
                   echo "Usage: $ProgName top"
                   echo ""
                   echo "   Get advanced information on the queue. Can be used to kill jobs."
                   echo ""

                   ;;
                *)
                    source $ChallengeHome/virtualenv/bin/activate
                    python $ChallengeHome/rlberry-farms/challenge/advanced_info.py
                    ;;
     esac
}


sub_egg(){
   sudo -u challenger cat $ChallengeHome/rlberry-farms/challenge/.egg.txt
}


subcommand=$1
case $subcommand in
    "" | "-h" | "--help")
        sub_help
        ;;
    *)
        shift
        sub_$subcommand $@
        ;;

esac
